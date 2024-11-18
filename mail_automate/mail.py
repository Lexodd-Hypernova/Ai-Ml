import sys
import pandas as pd
import smtplib
from email.message import EmailMessage
import re
from email.utils import formataddr
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
from google.oauth2 import service_account
from googleapiclient.discovery import build
from transformers import AutoModelForCausalLM, AutoTokenizer
def authenticate_google_apis(json_key_path):
    """Authenticate using service account JSON key for both Sheets and Drive."""
    credentials = service_account.Credentials.from_service_account_file(
        json_key_path, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
    )
    sheets_service = build('sheets', 'v4', credentials=credentials)
    return sheets_service

def read_sheet(sheets_service, spreadsheet_id, sheet_name):
    """Read data from a Google Sheet."""
    result = sheets_service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=sheet_name
    ).execute()
    values = result.get('values', [])
    max_cols = len(values[0]) if values else 0
    normalized_values = [row + [""] * (max_cols - len(row)) for row in values]
    df = pd.DataFrame(normalized_values[1:], columns=normalized_values[0])
    expected_columns = ["Email", "Name", "About", "Marked"]
    df = df.reindex(columns=expected_columns, fill_value="")
    return df
def write_sheet(sheets_service, spreadsheet_id, sheet_name, df):
    """Write data back to a Google Sheet."""
    values = [df.columns.tolist()] + df.values.tolist() 
    body = {
        'values': values
    }
    sheets_service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=sheet_name,
        valueInputOption='RAW',
        body=body
    ).execute()
def getLLMResponse(form_input, email_sender, email_recipient, email_style, Yourname):
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    tokenizer = AutoTokenizer.from_pretrained(model_name,device_map='cuda')
    model = AutoModelForCausalLM.from_pretrained(model_name,load_in_4bit=True,device_map='cuda')
    template = f"""
     Write a creative email in a {email_style} style on the topic: {form_input}. 
    The email is from Lexodd Hypernova Pvt. Ltd (email: {email_sender}) to {email_recipient} and recipient name is {Yourname}.
    Make sure the email is concise, does not repeat and include the input and include unnecessary commentary, placeholders, and ends naturally with 'Best regards, Lexodd Hypernova Pvt. Ltd'.After this there should not be anything
    Do not add any other analysis or notes about the structure of the email .
    """
    inputs = tokenizer(template, return_tensors="pt").to('cuda')
    outputs = model.generate(
        inputs['input_ids'],
        max_length=1024,
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
    
    msg = MIMEMultipart('related') 
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
    with open("Email_Banner.png", 'rb') as img_file:
        img_data = img_file.read()
    image = MIMEImage(img_data, name="Email_Banner.png")
    image.add_header('Content-ID', '<signature_image>')
    image.add_header('Content-Disposition', 'inline', filename="Email_Banner.png")
    msg.attach(image)
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
def main():
    json_key_path = "/content/omega-depot-441712-d0-3a691cc51db4.json"  
    spreadsheet_id = "1izWYqgiKyt2aIseALw0tpSUuA3IdV77OJrQLoOxlCKM"
    sheet_name = "Sheet1"
    pdf_path = sys.argv[1]  
    sheets_service = authenticate_google_apis(json_key_path)
    df = read_sheet(sheets_service, spreadsheet_id, sheet_name)
    print("Data retrieved from Google Sheets:")
    print(df.head())
    for i, j, k, l in zip(df["Email"], df["Name"], df["About"], df["Marked"].astype(bool)):
        if not l:
            send_llm_email(form_input=k, email_recipient=i, email_style="professional", name=j, pdf_path=pdf_path)
    df["Marked"] = True
    write_sheet(sheets_service, spreadsheet_id, sheet_name, df)
    print("Updated data written back to Google Sheets.")

if __name__ == "__main__":
    main()
