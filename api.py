from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("https://bitbucket.org/mohit_7781/project/raw/a42f6bc511e1f4ca99a90f563371a18a114fd64a/Data-for-countVectorizer.csv")
c = CountVectorizer(stop_words = 'english')
X_c = c.fit_transform(df['Text'])
print(X_c)


app = FastAPI()


@app.get("/")
async def root(item: str):
    
    input_txt = c.transform([item])
    loaded_model = pickle.load(open("Logistic.sav", 'rb'))
    ans = loaded_model.predict(input_txt)  
    print("*********************",ans)  
    print(type(ans[0]))
    return {"result":str(int(ans[0]))}
