import sys
import pandas as pd
import smtplib
from email.message import EmailMessage
import re
from email.utils import formataddr
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
<<<<<<< HEAD
from transformers.models.markuplm.modeling_markuplm import MarkupLMPredictionHeadTransform
import os
=======
from huggingface_hub import login
login()  #insert token name
>>>>>>> 839e090b7bd5f469d4d680725914c3bc8e0b9b94
def getLLMResponse(form_input, email_sender, email_recipient, email_style, Yourname):
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name,device_map='cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name,load_in_4bit=True,device_map='cuda')
    template = f"""
     Write a creative email in a {email_style} style on the topic: {form_input}. 
    The email is from Lexodd Hypernova Pvt. Ltd (email: {email_sender}) to {email_recipient} and recipient name is {Yourname}.
    Make sure the email is concise, does not repeat the input or include unnecessary commentary, placeholders, and ends naturally with 'Best regards, Lexodd Hypernova Pvt. Ltd'.
    Do not add any other analysis or notes about the structure of the email.
    """
    inputs = tokenizer(template, return_tensors="pt").to('cuda')
    outputs = model.generate(
        inputs['input_ids'],
        max_length=512,
        temperature=0.01,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def extract_email_components(llm_response):
    subject_match = re.search(r"Subject: (.+)\n", llm_response)
    subject = subject_match.group(1) if subject_match else "No Subject"
    body_start_index = llm_response.find("\n", subject_match.end()) if subject_match else 0
    body = llm_response[body_start_index:].strip()
    return subject, body

def send_llm_email(form_input, email_recipient, email_style, name, pdf_path=None):
    email_sender = "info@lexodd.com"
    llm_response = getLLMResponse(form_input, email_sender, email_recipient, email_style, name)
    subject, body = extract_email_components(llm_response)
    
    msg = MIMEMultipart('related')  # Use 'related' to handle both HTML and attachments properly
    msg['Subject'] = subject
    msg['From'] = formataddr(("Sender Name", email_sender))
    msg['To'] = email_recipient
    
    msg_alternative = MIMEMultipart('alternative')
    msg.attach(msg_alternative)
    msg_alternative.attach(MIMEText(body, 'plain'))
    
    html_content = """
    <html>
      <body>
        <p>{body}<br><br>
           Follow us on our LinkedIn:<br>
           LinkedIn: <a href="https://www.linkedin.com/company/lexodd-hypernova/">LinkedIn</a><br>
           <img src="cid:signature_image" alt="Lexodd Signature" style="width:900px;"><br>
        </p>
      </body>
    </html>
    """.format(body=body.replace('\n', '<br>'))
    msg_alternative.attach(MIMEText(html_content, 'html'))
    
    # Attach the banner image
    with open("Email_Banner.png", 'rb') as img_file:
        img_data = img_file.read()
    image = MIMEImage(img_data, name="Email_Banner.png")
    image.add_header('Content-ID', '<signature_image>')
    image.add_header('Content-Disposition', 'inline', filename="Email_Banner.png")
    msg.attach(image)
    
    # Optionally attach a PDF if a path is provided
    if pdf_path:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_data = pdf_file.read()
        pdf_attachment = MIMEBase('application', 'pdf')
        pdf_attachment.set_payload(pdf_data)
        encoders.encode_base64(pdf_attachment)
        pdf_attachment.add_header('Content-Disposition', 'attachment', filename=pdf_path.split('/')[-1])
        msg.attach(pdf_attachment)
    
    smtp_server = 'smtp.gmail.com'
    smtp_port = 465
    sender_email = 'info@lexodd.com'
    sender_password = 'ncnr sjwd argp zjon'
    
    try:
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)

        print(f"Email sent to {email_recipient} with subject: {subject}")
    except Exception as e:
        print(f"Failed to send email: {e}")

def csv_mail():
    csv_path=sys.argv[1]
    df=pd.read_excel(csv_path)
    email_recipient=df["Email"]
    email_name=df["Name"]
    email_subject=df["About"]
    marked=df["Marked"]
    return list(email_recipient),list(email_name),list(email_subject),list(marked)

def main():
    email_recipient,email_name,email_subject,marked=csv_mail()
    pdf_path = sys.argv[2]  
    for i,j,k,l in zip(email_recipient,email_name,email_subject,marked):
      if l!=True:
        send_llm_email(
          form_input=k,
          email_recipient=i,
          email_style="professional",
          name=j,
          pdf_path=pdf_path  
      )
<<<<<<< HEAD
    df=pd.read_excel(sys.argv[1])
    Marked=[True]*(df.shape[0])
    df["Marked"]=Marked
    writer = pd.ExcelWriter(path=f'{os.getcwd()}/Lexodd.xlsx',engine='xlsxwriter')
    df.to_excel(index=False,excel_writer=writer)
    writer.close()

main()
#how to run
#!python /content/mail.py  /content/LexoddContactSheet.xlsx /content/NTT.pdf
=======
main()
>>>>>>> 839e090b7bd5f469d4d680725914c3bc8e0b9b94
