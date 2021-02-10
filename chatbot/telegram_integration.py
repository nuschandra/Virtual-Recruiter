import random
import requests
import json

def bot_initialize(user_msg):
    flag=True
    while(flag==True):
        user_response = user_msg
        if(user_response not in bye):
            if(user_response == '/start'):
                bot_resp = """Hi! There. I am your Corona Protector. I can tell you all the Facts and Figures, Signs and Symptoms related to spread of Covid-19 in India. \nType Bye to Exit.""" 
                return bot_resp
            elif(user_response in thank_you):
                bot_resp = random.choice(thank_response)
                return bot_resp
            elif(user_response in greetings):
                bot_resp = random.choice(greetings) + ", What information you what related to Covid-19 in India"
                return bot_resp
            else:
                user_response = user_response.lower()
                bot_resp = response(user_response)
                sent_tokens.remove(user_response)   # remove user question from sent_token that we added in sent_token in response() to find the Tf-Idf and cosine_similarity
                return bot_resp
        else:
            flag = False
            bot_resp = random.choice(bye)
            return bot_resp


class telegram_bot():
    def __init__(self):
        self.token = "1537657914:AAEspo0IA7tiW2CCAnWLfsxOd0YabGC-r50"
        self.url = f"https://api.telegram.org/bot{self.token}"
    def get_updates(self,offset=None):
        url = self.url+"/getUpdates?timeout=100"
        if offset:
            url = url+f"&offset={offset+1}"
        url_info = requests.get(url)
        return json.loads(url_info.content)
    def send_message(self,msg,chat_id):
        url = self.url + f"/sendMessage?chat_id={chat_id}&text={msg}"
        if msg is not None:
            requests.get(url)
    def grab_token(self):
        return tokens

tbot = telegram_bot()
update_id = None
def get_response(msg):     
    if msg is not None:
        reply = bot_initialize(msg)     
    return reply
       
while True:
    updates = tbot.get_updates(offset=update_id)
    updates = updates['result']
    print(updates)
    if updates:
        for item in updates:
            update_id = item["update_id"]
            print(update_id)
            try:
                message = item["message"]["text"]
                print(message)
            except:
                message = None
            from_ = item["message"]["from"]["id"]
            print(from_)
        reply = get_response(message)
        tbot.send_message(reply,from_)