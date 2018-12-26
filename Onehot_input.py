
# coding: utf-8

# In[2]:


import midi
pattern = midi.read_midifile('./midi/Frog.mid')
re = pattern.resolution
print(re)
#全,二分,四分,2拍3連,8分,3連符,16分,5連,6連,32分,64分
t = [re*4,re*2,re,int(re*2/3),int(re/2),int(re/3),int(re/4),int(re/5),int(re/6),int(re/8),int(re/16)] 
print(t)


# In[47]:


def makeonehot_pitch(pp,f):
    for l in range(0, 128): 
        if(pp == l):
            f.write("1")
        else:
            f.write("0")

def makeonehot_tick(tt,f):
    if(tt > sum(t)):
        tt = t[0]
    for i in t:
        if(tt >= i):
            f.write("1")
            tt -= i
        else:
            f.write("0")


# In[57]:


f = open("input_vector.txt","w")
print("vero\ttick\tpitch")
for i, chunk in enumerate(pattern[0]):
    chunk_str = ""
    if(chunk.name == "Note On"):
        tt = chunk.tick
        print(str(chunk.velocity)+"\t"+str(tt)+"\t"+str(chunk.pitch))
        if(chunk.velocity > 0): # velocityは強弱、tickの値は直前の休符の長さ、pitchの値は音の高さ
            if(chunk.tick != 0):
                makeonehot_pitch(0,f) # 休符なので128文字0を打つ
                f.write("1") #音出してないので1
                makeonehot_tick(tt,f)
                f.write("\n")
            pp = chunk.pitch
        else:
            if(tt == 0):
                print("error!")
                break
            makeonehot_pitch(pp,f)
            f.write("0") # 音出してるので0
            makeonehot_tick(tt,f)
            f.write("\n")
f.close()

