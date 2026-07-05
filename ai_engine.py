"""HF inference wrapper."""
import os,requests
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
URL=f"https://api-inference.huggingface.co/models/{MODEL}"

def generate(system_prompt,user_prompt,max_tokens=500,temp=0.2):
 headers={"Content-Type":"application/json"}
 tok=os.getenv("HF_API_KEY","")
 if tok: headers["Authorization"]=f"Bearer {tok}"
 payload={"inputs":f"System:\n{system_prompt}\n\nUser:\n{user_prompt}\nAssistant:","parameters":{"max_new_tokens":max_tokens,"temperature":temp,"return_full_text":False},"options":{"wait_for_model":True}}
 r=requests.post(URL,headers=headers,json=payload,timeout=180)
 r.raise_for_status();j=r.json();
 if isinstance(j,list): return j[0].get("generated_text","")
 return j.get("generated_text",str(j))
