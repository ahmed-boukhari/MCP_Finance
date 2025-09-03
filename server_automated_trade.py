#!/usr/bin/env python3
import json,re,time,functools,logging,os,sys
from typing import Dict,List,Any,Tuple,Optional
from datetime import datetime
import numpy as np
os.environ.update({'NUMBA_DISABLE_INTEL_SVML':'1','NUMBA_DISABLE_HSA':'1','NUMBA_DISABLE_CUDA':'1','NUMBA_THREADING_LAYER':'safe'})

# Ultra-compact imports with fallbacks
try:
    import numba
    from numba import njit
    N=1
except ImportError:
    N=0

try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import nltk
    for x in ['punkt','vader_lexicon','stopwords','averaged_perceptron_tagger','wordnet']:
        if not os.path.exists(os.path.join(nltk.data.path[0] if nltk.data.path else '', f'tokenizers/{x}' if 'punkt' in x else f'corpora/{x}')):
            nltk.download(x,quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    NLP=1
    sw=set(stopwords.words('english'))
    lm=WordNetLemmatizer()
except ImportError:
    NLP=0
    sw=set()
    lm=None

try:
    import spacy
    nlp_model=spacy.load('en_core_web_sm')
    SP=1
except ImportError:
    SP=0
    nlp_model=None

import yfinance as yf
import v20
from mcp.server.fastmcp import FastMCP

# LLVM-optimized functions
if N:
    @njit(cache=1,nogil=1)
    def vi(c):
        return len(c)>6 and c[3]==95 and all(65<=c[i]<=90 for i in range(3)) and all(65<=c[i]<=90 for i in range(4,min(7,len(c))))
    
    @njit(cache=1,nogil=1)
    def fi(c):
        return np.array([x-32 if 97<=x<=122 else 95 if x==47 else x for x in c])
    
    @njit(cache=1,nogil=1)
    def vu(u):
        return (0,1) if u==0 else (0,2) if abs(u)<1 else (0,3) if abs(u)>1e7 else (1,0)
    
    @njit(cache=1,nogil=1)
    def cm(u,p):
        return abs(u)*p*.02
    
    @njit(cache=1,nogil=1)
    def bv(ua,pa,mm):
        vf=np.zeros(len(ua),dtype=np.bool_)
        tm=0.
        for i in range(len(ua)):
            if vu(ua[i])[0] and tm+cm(ua[i],pa[i])<=mm:
                vf[i]=1
                tm+=cm(ua[i],pa[i])
        return vf
    
    @njit(cache=1,nogil=1)
    def csn(v1,v2):
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
        return np.dot(v1,v2)/(n1*n2) if n1>0 and n2>0 else 0.0
else:
    def vi(c):
        code_str=''.join(chr(int(x)) for x in c)[:7]
        return '_' in code_str and len(code_str.split('_')[0])==3 and len(code_str.split('_')[1])==3 if len(c)>6 else 0
    
    def fi(c):
        return np.array([x-32 if 97<=x<=122 else 95 if x==47 else x for x in c])
    
    def vu(u):
        return (0,1) if u==0 else (0,2) if abs(u)<1 else (0,3) if abs(u)>1e7 else (1,0)
    
    def cm(u,p):
        return abs(u)*p*.02
    
    def bv(ua,pa,mm):
        return np.array([vu(ua[i])[0] and sum(cm(ua[j],pa[j]) for j in range(i+1))<=mm for i in range(len(ua))])
    
    def csn(v1,v2):
        n1,n2=np.linalg.norm(v1),np.linalg.norm(v2)
        return np.dot(v1,v2)/(n1*n2) if n1>0 and n2>0 else 0.0

class NM:
    def __init__(s):
        # Major forex pairs
        s.fx=['EUR_USD','GBP_USD','USD_JPY','USD_CHF','USD_CAD','AUD_USD','NZD_USD','EUR_GBP','EUR_JPY','GBP_JPY','CHF_JPY','CAD_JPY','AUD_JPY','NZD_JPY','EUR_CHF','EUR_CAD','EUR_AUD','EUR_NZD','GBP_CHF','GBP_CAD','GBP_AUD','GBP_NZD','USD_SGD','USD_HKD','USD_ZAR','USD_MXN','USD_NOK','USD_SEK','USD_DKK','USD_PLN','USD_CZK','USD_HUF','USD_TRY']
        
        # Country/region semantic vectors (simplified embeddings)
        s.cv={
            'USD':np.array([1.0,0.8,0.9,0.7,0.6]),
            'EUR':np.array([0.9,1.0,0.8,0.6,0.7]),
            'GBP':np.array([0.8,0.7,1.0,0.5,0.6]),
            'JPY':np.array([0.7,0.6,0.5,1.0,0.8]),
            'CHF':np.array([0.6,0.8,0.6,0.7,1.0]),
            'CAD':np.array([0.9,0.5,0.6,0.4,0.5]),
            'AUD':np.array([0.5,0.4,0.8,0.6,0.4]),
            'NZD':np.array([0.4,0.3,0.7,0.5,0.3])
        }
        
        # Economic concept vectors
        s.ec={
            'inflation':np.array([0.8,0.2,0.9,0.7,0.6]),
            'gdp':np.array([0.9,0.8,0.7,0.8,0.9]),
            'employment':np.array([0.7,0.6,0.8,0.5,0.7]),
            'trade':np.array([0.6,0.9,0.6,0.8,0.7]),
            'monetary':np.array([0.9,0.7,0.8,0.9,0.8]),
            'fiscal':np.array([0.8,0.8,0.7,0.6,0.9]),
            'oil':np.array([0.7,0.5,0.4,0.3,0.8]),
            'gold':np.array([0.6,0.4,0.3,0.2,0.9]),
            'tech':np.array([0.9,0.8,0.9,0.8,0.6])
        }
        
        # Currency strength patterns
        s.cp={
            'USD':['dollar','fed','federal','reserve','powell','yellen','treasury','biden','trump','america','american','us','united','states'],
            'EUR':['euro','ecb','lagarde','draghi','europe','european','germany','german','france','french','italy','italian','spain','spanish'],
            'GBP':['pound','sterling','boe','bailey','carney','britain','british','uk','england','english','london'],
            'JPY':['yen','boj','kuroda','japan','japanese','tokyo','nikkei','sony','toyota'],
            'CHF':['franc','snb','swiss','switzerland','zurich','geneva'],
            'CAD':['canadian','canada','boc','poloz','macklem','toronto','oil','crude'],
            'AUD':['australian','australia','rba','lowe','stevens','sydney','melbourne','commodity','iron','coal'],
            'NZD':['zealand','kiwi','rbnz','orr','wheeler','auckland','wellington']
        }
        
        # Sector-to-currency mapping weights
        s.sw={
            'technology':{'USD':0.9,'EUR':0.7,'GBP':0.6,'JPY':0.8},
            'energy':{'USD':0.8,'CAD':0.9,'AUD':0.7,'NOK':0.8},
            'finance':{'USD':0.9,'GBP':0.8,'EUR':0.7,'CHF':0.8},
            'commodity':{'AUD':0.9,'CAD':0.8,'NZD':0.7,'ZAR':0.8},
            'automotive':{'JPY':0.9,'EUR':0.8,'USD':0.7},
            'luxury':{'EUR':0.8,'CHF':0.9,'GBP':0.7}
        }

class MI:
    def __init__(s):
        s.nm=NM()
        s.va=SentimentIntensityAnalyzer() if NLP else None
        s.k={'b':['growth','profit','revenue','earnings','beat','outperform','strong','positive','bull','rally','surge','soar','climb','upgrade','overweight','buy','momentum','breakthrough'],'br':['decline','loss','recession','bearish','fall','drop','weak','negative','bear','crash','plunge','tumble','downgrade','underweight','sell','concern','warning'],'vol':['volatile','uncertainty','risk','fluctuation','swing','instability','turbulence','unpredictable','erratic']}

    def ex(s,tt:str)->List[str]:
        """Extract entities using multiple NLP techniques"""
        ents=[]
        
        # Stock symbols
        ents.extend(re.findall(r'\$([A-Z]{1,5})',tt.upper()))
        
        # Currency codes
        ents.extend(re.findall(r'\b([A-Z]{3})\b',tt))
        
        # Company mentions
        ents.extend(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',tt))
        
        if SP and nlp_model:
            doc=nlp_model(tt)
            for ent in doc.ents:
                if ent.label_ in ['ORG','GPE','PERSON','MONEY']:
                    ents.append(ent.text)
        
        return list(set(ents))

    def tv(s,tt:str)->np.ndarray:
        """Generate semantic vector for text"""
        if not NLP:
            return np.random.rand(5)*.1
        
        tb=TextBlob(tt.lower())
        words=[lm.lemmatize(w) for w in tb.words if w not in sw and len(w)>2] if lm else tt.lower().split()
        
        vec=np.zeros(5)
        wc=0
        
        for w in words:
            for cc,cv in s.nm.cv.items():
                if any(kw in w for kw in s.nm.cp.get(cc[:2]if len(cc)>2 else cc,[])):
                    vec+=cv
                    wc+=1
            
            for ec,ev in s.nm.ec.items():
                if ec in w or w in ec:
                    vec+=ev*.5
                    wc+=1
        
        return vec/max(wc,1)

    def cm(s,tt:str)->Tuple[List[str],List[float]]:
        """Map text to currencies with confidence scores"""
        tv=s.tv(tt)
        ents=s.ex(tt)
        cs=[]
        
        # Direct currency mentions
        curs=set()
        for e in ents:
            if e.upper() in ['USD','EUR','GBP','JPY','CHF','CAD','AUD','NZD']:
                curs.add(e.upper())
        
        if not curs:
            # Semantic matching
            scores={}
            for cc,cv in s.nm.cv.items():
                scores[cc]=csn(tv,cv)
            
            # Get top currencies
            top_curs=sorted(scores.items(),key=lambda x:x[1],reverse=True)[:3]
            curs=[c for c,sc in top_curs if sc>.3]
            cs=[sc for c,sc in top_curs if sc>.3]
        
        return list(curs),cs if cs else [.8]*len(curs)

    def fp(s,curs:List[str])->str:
        """Select best forex pair from currencies"""
        if len(curs)<2:
            curs.extend(['USD','EUR'][:2-len(curs)])
        
        # Priority pairs (most liquid)
        pp=['EUR_USD','GBP_USD','USD_JPY','USD_CHF','USD_CAD','AUD_USD']
        
        # Check all priority pairs first
        for p in pp:
            c1, c2 = p.split('_')
            if c1 in curs and c2 in curs:
                return p
        
        # Check all possible combinations from detected currencies
        for i in range(len(curs)):
            for j in range(i + 1, len(curs)):
                c1, c2 = curs[i], curs[j]
                # Try both directions for each pair
                for fmt in [f'{c1}_{c2}', f'{c2}_{c1}']:
                    if fmt in s.nm.fx:
                        return fmt
        
        # If only one currency detected, try pairing with major currencies in order
        if len(curs) == 1:
            major_curs = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD']
            detected = curs[0]
            
            for major in major_curs:
                if major != detected:
                    for fmt in [f'{detected}_{major}', f'{major}_{detected}']:
                        if fmt in s.nm.fx:
                            return fmt
        
        # Enhanced fallback: prioritize pairs containing detected currencies
        if curs:
            for c in curs:
                for pair in s.nm.fx:
                    if c in pair:
                        return pair
        
        # Ultimate fallback
        return 'EUR_USD'

    def atw(s,tt:str,tck:str=None)->Tuple[Dict[str,Any],str]:
        """Analyze tweet with NLP-based forex mapping"""
        try:
            # Sentiment analysis
            if NLP and s.va:
                vs=s.va.polarity_scores(tt)
                sc=vs['compound']
                conf=max(abs(vs['pos']-vs['neg']),0.1)
            else:
                # Fallback keyword-based sentiment
                tl=tt.lower()
                bl=sum(kw in tl for kw in s.k['b'])
                br=sum(kw in tl for kw in s.k['br'])
                sc=(bl-br)/max(bl+br,1)
                conf=min(abs(sc)*2,1) if sc!=0 else 0.1
            
            # Currency mapping
            curs,c_conf=s.cm(tt)
            fx_pair=s.fp(curs)
            
            # Trading signals
            if sc>.3:
                sig,act='STRONG_BUY','buy'
            elif sc>.1:
                sig,act='BUY','buy'
            elif sc<-.3:
                sig,act='STRONG_SELL','sell'
            elif sc<-.1:
                sig,act='SELL','sell'
            else:
                sig,act='HOLD','hold'
            
            # Extract ticker if mentioned
            if not tck:
                tcks=re.findall(r'\$([A-Z]{1,5})',tt.upper())
                tck=tcks[0] if tcks else None
            
            res={
                'ticker':tck,
                'sentiment':'bullish' if sc>0 else 'bearish' if sc<0 else 'neutral',
                'score':round(sc,3),
                'confidence':round(conf*np.mean(c_conf) if c_conf else conf,3),
                'signal':sig,
                'action':act,
                'currencies':curs,
                'currency_confidence':c_conf,
                'extracted_entities':s.ex(tt)[:10],  # Limit output
                'timestamp':int(time.time()),
                'nlp_enabled':NLP,
                'spacy_enabled':SP
            }
            
            return res,fx_pair
            
        except Exception as e:
            return {'error':str(e),'fallback_pair':'EUR_USD'},'EUR_USD'

class TE:
    def __init__(s):
        s.ps={}

    async def cmo(s,ai,tk,i,u,hn="api-fxpractice.oanda.com",pt=443,ep=1.):
        try:
            uf=float(u)
            iv,ec=vu(uf)
            if not iv:
                return json.dumps({"success":0,"error":{1:"Units cannot be zero",2:"Units too small",3:"Units too large"}[ec],"llvm_enabled":N})
            
            mr=cm(uf,ep)
            api=v20.Context(hn,pt,token=tk)
            r=api.order.market(ai,instrument=i,units=u)
            
            res={"success":r.status==201,"status":r.status,"instrument":i,"units":u,"margin_required":round(mr,2),"llvm_enabled":N}
            
            if hasattr(r,'body') and r.body:
                for a in ['orderCreateTransaction','orderFillTransaction','lastTransactionID']:
                    if hasattr(r.body,a):
                        attr_val = getattr(r.body,a)
                        res[a] = attr_val.dict() if hasattr(attr_val,'dict') else str(attr_val)
            
            return json.dumps(res,indent=2,default=str)
        except Exception as e:
            return json.dumps({"success":0,"error":str(e),"llvm_enabled":N})
        
    async def bvo(s,ords,mtm=1e5):
        try:
            if not ords:
                return json.dumps({"valid_orders":[],"total_margin":0})
            
            ua=np.array([float(o.get('units',0)) for o in ords])
            pa=np.array([float(o.get('estimated_price',1)) for o in ords])
            vf=bv(ua,pa,mtm)
            vo=[]
            tm=0
            
            for i,(o,iv) in enumerate(zip(ords,vf)):
                if iv:
                    m=cm(ua[i],pa[i])
                    vo.append({**o,"index":i,"margin_required":round(m,2)})
                    tm+=m
            
            return json.dumps({"valid_orders":vo,"total_submitted":len(ords),"valid_count":len(vo),"total_margin":round(tm,2),"margin_utilization":round(tm/mtm*100,2),"llvm_enabled":N})
        except Exception as e:
            return json.dumps({"error":str(e),"llvm_enabled":N})
    
    def gps(s):
        return{"llvm_enabled":N,"cache_size":len(s.ps),"nlp_enabled":NLP,"spacy_enabled":SP}

te=TE()
mi=MI()
server=FastMCP("oanda-ultra-nlp",instructions=f"Ultra-Compact OANDA Trading with Advanced NLP Forex Mapping - {'LLVM' if N else 'STD'}+{'NLP' if NLP else 'Basic'}+{'SpaCy' if SP else ''}")

@server.tool("automated_trade")
async def at(ai:str,tk:str,tck:str='',tt:str='',mu:int=5000,mc:float=.6,hn:str="api-fxpractice.oanda.com",vo:bool=False)->str:
    try:
        if not tt:
            return json.dumps({'error':'Tweet text required for NLP analysis'})
        
        a,i=mi.atw(tt,tck)
        
        if 'error' in a:
            return json.dumps({'error':a['error'],'fallback_used':True})
        
        if a['confidence']<mc:
            return json.dumps({
                'action':'no_trade',
                'reason':f"Confidence {a['confidence']:.2f} < {mc}",
                'analysis':a,
                'suggested_pair':i
            })
        
        u=str(int(min(mu*a['confidence'],mu)*(1 if a['sentiment']=='bullish' else -1)))
        
        if vo:
            return json.dumps({
                'validation':await te.bvo([{'units':u,'estimated_price':1}],1e5),
                'analysis':a,
                'proposed_trade':{'instrument':i,'units':u}
            },indent=2)
        else:
            return json.dumps({
                'analysis':a,
                'trade':{
                    'instrument':i,
                    'units':u,
                    'confidence':a['confidence'],
                    'result':await te.cmo(ai,tk,i,u,hn,443,1)
                },
                'timestamp':datetime.now().isoformat()
            },indent=2)
    except Exception as e:
        return json.dumps({'error':str(e)})

@server.tool("analyze_text_to_forex")
async def atf(tt:str,dt:bool=True)->str:
    """Analyze any text and map to forex pairs using advanced NLP"""
    try:
        a,fp=mi.atw(tt)
        
        if dt:
            return json.dumps({
                'text_analysis':a,
                'recommended_pair':fp,
                'all_major_pairs':mi.nm.fx[:10],  # Show top 10
                'mapping_method':'semantic_vector_analysis' if NLP else 'keyword_fallback',
                'processing_time':time.time()
            },indent=2)
        else:
            return json.dumps({
                'pair':fp,
                'sentiment':a.get('sentiment','neutral'),
                'confidence':a.get('confidence',0),
                'currencies':a.get('currencies',[])
            })
    except Exception as e:
        return json.dumps({'error':str(e)})

@server.tool("system_info")
async def si(dt:bool=False)->str:
    try:
        if dt:
            return json.dumps({
                "performance":te.gps(),
                "capabilities":{
                    "optimized_trading":N,
                    "sentiment_analysis":NLP,
                    "spacy_ner":SP,
                    "semantic_mapping":1,
                    "entity_extraction":1,
                    "forex_pairs_supported":len(mi.nm.fx)
                },
                "forex_pairs":mi.nm.fx,
                "supported_currencies":list(mi.nm.cv.keys())
            },indent=2,default=str)
        else:
            return json.dumps({
                "tools":{
                    "automated_trade":"NLP-powered tweet-to-forex trading",
                    "analyze_text_to_forex":"Map any text to forex pairs",
                    "create_market_order":"Execute OANDA trades",
                    "batch_validate_orders":"Validate multiple orders",
                    "system_info":"Enhanced system status"
                },
                "optimization":f"{'LLVM' if N else 'STD'} + {'Advanced NLP' if NLP else 'Basic'} + {'SpaCy' if SP else ''}",
                "forex_pairs":len(mi.nm.fx)
            },indent=2,default=str)
    except Exception as e:
        return json.dumps({"error":str(e)})

def main():
    print(f"Enhanced OANDA NLP Server | {'LLVM' if N else 'STD'}+{'NLP' if NLP else 'Basic'}+{'SpaCy' if SP else ''} | {len(mi.nm.fx)} forex pairs | Advanced semantic mapping")
    server.run(transport="stdio")

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)