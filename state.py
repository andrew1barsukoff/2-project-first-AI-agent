from typing import TypedDict

class UserState(TypedDict):
    count:int
    msg:str
    human_state:str
    whose_answer:str
    const_token: int(600)
    const_reserve : int(100)
    def __init__(x):
        return {'count':x}
        
