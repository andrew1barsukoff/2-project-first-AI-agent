import os
from langchain_gigachat.chat_models import GigaChat
from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage,  AIMessage
from langgraph.graph import StateGraph, END
from nltk.corpus import stopwords
import re
from state import UserState
stop_words=stopwords.words('russian')
load_dotenv(find_dotenv('.env.txt',usecwd=True)) #API file
'''
You can use absolutely any other neural network
'''
llm_giga = GigaChat(
   credentials=os.environ['API_SBER_KEY'],
   scope="GIGACHAT_API_PERS",
   model="GigaChat-2",
)

messages = []
    
'''
I performed lemmatization so that only the 
most important tokens remain in memory; this way, we save memory space.
'''
def lemmatizer(state:UserState):
    words_sub=re.sub("[^\w\s]+[\d]+",' ',state['msg'])
    words_eng_sub=re.sub("[a-zA-Z\s]+",' ',words_sub)
    words_to_vekt=[]
    for word in words_eng_sub.split():
         if word in stop_words:
             continue
         else:
             words_to_vekt.append(word)
    words_to_vekt=' '.join(words_to_vekt)
    return {'msg':words_to_vekt}

def add_in_messages(state:UserState):
    if state["whose_answer"]=='II':
        messages.append(AIMessage(content=state['msg']))
    elif state["whose_answer"]=='human':
        messages.append(HumanMessage(content=state['msg']))
        
def messages_for_ii(state:UserState):
    mst = input("Введите свое сообщение:")
    return {
        'msg':mst,
        'whose_answer':'human'
    }

def check_mesage(state:UserState):
    if (state["msg"]=='пока'):
        return {'human_state':'end'}
    else:
        return {'human_state':'go'}

def ii_answer(state:UserState):
    ans=llm_giga.invoke(messages)
    print(ans.content)
    return {
        'msg':ans.content,
        'whose_answer':'II'
    }

def begin_branch(state:UserState):
    if state['whose_answer']=='II':
        return '1'
    else:
        return '0'

def should_continue(state:UserState):
    if state['human_state']=='end':
        return '0'
    elif state['human_state']=='go' and state['whose_answer']=='human':
        return '2'
    else:
        return '1'
 
def count_token(state: UserState):
    y=state.get("count",0)+len((state["msg"]).split())
    return {'count':y} 
     
def compare_token(state:UserState):
    if state.get('const_token',600)-state.get('const_reserve',100)<state['count']:
        return '0'
    else:
        return '1' 
    
build = StateGraph(UserState)
build.add_node("messages_for_ii", messages_for_ii)
build.add_node("ii_answer", ii_answer)
build.add_node("check_mesage", check_mesage)
build.add_node("lemmatizer", lemmatizer)
build.add_node("count_token", count_token)
build.add_node("add_in_messages", add_in_messages)


build.set_conditional_entry_point(begin_branch, {"1": "messages_for_ii", "0": "ii_answer"})
build.add_edge("messages_for_ii", 'check_mesage') 
build.add_edge("check_mesage", 'lemmatizer') 
build.add_edge("ii_answer", 'lemmatizer') 
build.add_edge("lemmatizer", 'count_token') 
build.add_edge("count_token", 'add_in_messages')
build.add_conditional_edges(    
    "add_in_messages",
    should_continue,
    {
        "1": "messages_for_ii",  
        "0": END,   
        '2': "ii_answer",
    }
)


graph = build.compile()
graph.invoke({"whose_answer":'II'})
    
    
























      
          




