import time

import pandas as pd
from openai import OpenAI

# file_path = R'dataset-single_domain\Food\traindata_new.txt'
file_path = R'dataset-single_domain\Food\list.txt'
# df = pd.read_csv(file_path, sep='\t', header=None)

# 逐行读取文件
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 处理每一行
data = []
for line in lines:
    row = line.strip().split('\t')  # 按制表符分割
    data.append(row)

# 转换为DataFrame
column = ['id', 'asin', 'index']
df_item = pd.DataFrame(data, columns=column)
df_item.set_index('index', inplace=True)

# client = OpenAI(
#     base_url='https://api.nuwaapi.com/v1',
#     api_key=''
# )

client = OpenAI(
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key=''
)

food_path = "../dataset/meta_Grocery_and_Gourmet_Food.json.gz"
kitchen_path = '../dataset/meta_Home_and_Kitchen.json.gz'

movie_path = '../dataset/meta_Movies_and_TV.json.gz'
book_path = '../dataset/meta_Books.json.gz'

sport_path = '../dataset/meta_Sports_and_Outdoors.json.gz'
cloth_path = '../dataset/meta_Clothing_Shoes_and_Jewelry.json.gz'

paths = [food_path, kitchen_path, movie_path, book_path, sport_path, cloth_path]
domains = ['Food', 'Kitchen', 'Movie', 'Book', 'Sport', 'Clothing']

df_food = pd.read_json(food_path, lines=True)
df_food.drop_duplicates(subset=['asin'], inplace=True)


# df_kitchen = pd.read_json(kitchen_path, lines=True)

def get_embedding(df, asin):
    # 查询特定的asin
    item = df[df["asin"] == asin]

    if not item.empty:
        title = item['title'].item()
        category = item['category'].item()
        description = item['description'].item()
        text_desc = f'title: {title}\ncategory: {category}\ndescription: {description}'
        print(len(text_desc))
        response = client.embeddings.create(
            input=text_desc,
            model="text-embedding-v3",
            dimensions=1024,
            encoding_format="float"
        )

        return response.data[0].embedding
    else:
        return [0] * 1024


df_item['embedding'] = df_item['asin'].apply(lambda x: get_embedding(df_food, x))
# embeddings = []
# for index, series in df_item.iterrows():
#     embedding = get_embedding(df_food, series['asin'])
#     embeddings.append([series['id'], series['asin'], embedding])
#     # time.sleep(3)
#     if int(index) % 100 == 0:
#         pd.DataFrame(embeddings, columns=['id', 'asin', 'embedding']).to_csv(f'dataset/food_embeddings{index}.csv')
df_item.to_csv('dataset/food_embeddings.csv')
