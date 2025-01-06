from openai import OpenAI

client = OpenAI(api_key='123', base_url='http://0.0.0.0:23333/v1')
model_name = client.models.list().data[0].id
response = client.chat.completions.create(
    model=model_name,
    messages=[{
        'role':
        'user',
        'content': [{
            'type': 'text',
            'text': 'hello',
        }],
    }],
    temperature=0.8,
    top_p=0.8,
    max_tokens=256)
print(response)