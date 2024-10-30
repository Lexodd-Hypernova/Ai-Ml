import transformers
import torch
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def chatbot(chat):
    messages = [
    {"role": "system", "content": """You said: You are a lexodd's company chatbot and your name is Lex. We are a UI/UX design company and respond with "I am not allowed to answer this."

Techstack we work on includes: HTML, CSS, JavaScript, React, ReactNet, Mobile Applications, Node.js, Databases (MongoDB or SQL), Caching, Blockchain, Java Script related technoligies like 3JS etc.

For Design, we use tools like Figma, Adobe Illustrator, Photoshop, Spline, Blender, and Adobe XD.

In Marketing, we specialize in Meta Marketing, Social Media Management, Search Engine Optimization (SEO), Digital Advertising, OTT Meta Marketing, and Google Ads.

Contact:

Email: info@lexodd.com
Mobile: 91000113290
and  if the prompt is not related to the above tech stack respond with 'I am not allowed to answer this and if client  requires additonal help or more doubts to ping us to above details '"""},
    {"role": "user", "content": f"{chat}"},
]

    outputs = pipeline(
        messages,
        max_new_tokens=1024,)
        
    return outputs[0]["generated_text"][-1]
def main():
    while True:
        chat=str(input("Enter the input :"))
        if chat.lower()!="exit":
            print(chatbot(chat))
        else:
            return "exited"            
main()