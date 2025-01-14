from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import login, authenticate
from pypdf import PdfReader
from dotenv import load_dotenv
from .models import Profile, UploadedPDF, UploadedCsv
import openai
import json
import os
from django.http import JsonResponse
from .forms import PDFUploadForm, UploadedImageForm, UploadedImageFile
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import base64
import re
import textwrap
import pandas as pd

# Load environment variables at startup
load_dotenv()

text = "Please perform information extraction on the following text. Return the results as a JSON object with keys for 'Full_Name', 'Email', 'Total_years_of_Experience', 'Name_Company_Experience', 'Last_level_of_Education', 'phone_number', 'location', 'skills'(skills separated by commas), and 'Description' of the person. Ensure key names match the information types provided."


def index(request):
    if request.method == 'POST':
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
        if 'cv'in request.POST:
            resume = request.FILES['resume']
            reader = PdfReader(resume)
            page = reader.pages[0]
            fullContent = ''
            for i in range(len(reader.pages)):
                page = reader.pages[i]
                fullContent += page.extract_text()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to extract data as output JSON."},
                    {"role": "user", "content":text + " " + fullContent }
                ]
            )
            if(response):
                data = json.loads(response.choices[0].message.content)
                contexte = {
                    'data':data,
                }
                return render(request, 'signup.html', contexte)
            else:
                return render(request, 'index.html')
        
    if 'buildCareer' in request.POST:
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        if password1 != password2:
            return False    
        if User.objects.filter(email=email).exists():
            return render(request, 'setpassword.html', {'error': 'User with this email already exists'})
        user = User.objects.create_user(username=email, email=email, password=password1)
        user = authenticate(username=email, password=password1)
        login(request, user)
        profile = Profile.objects.create(user=user,name = request.POST.get('fullName'),email = email,phone_number = request.POST.get('phone'),location = request.POST.get('location'),description = request.POST.get('description'),level_education = request.POST.get('education'),last_company = request.POST.get('lexperience'),year_of_experience = request.POST.get('experience'),desired_job = request.POST.get('job'),desired_location = request.POST.get('locationJob'),skills = request.POST.get('skills'))
        profile.save()
        return render(request, 'index.html')
    if 'login' in request.POST:
        email = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=email, password=password)
        if user is not None:
            login(request, user)
            if user.is_staff:
                return redirect('dashboard')
            else:
                return render(request, 'jobs.html')
        else:
            return render(request, 'index.html', {'error':'Username or Password not exist'})

    else:                
        return render(request, 'index.html')

def dashboard(request):
    return render(request, 'dashboard.html')

def chatpdf(request):
    load_dotenv()
    form = PDFUploadForm()
    if request.method == 'POST':
        if 'uploadpdf' in request.POST:
            form = PDFUploadForm(request.POST, request.FILES)
            if form.is_valid():
                uploaded_pdf = form.save()
                pdf = uploaded_pdf.pdf_file.path
                pdf_id = uploaded_pdf.id 
                request.session['uploaded_pdf_id'] =   pdf_id
                return JsonResponse({'pdf_id': pdf_id, 'message':"You've successfully uploaded the file. Feel free to ask anything you'd like"}, status=400)
            else:
                return JsonResponse({'message': 'Error uploading file, Try again'}, status=400)
          
        if 'message' in request.POST:
            message = request.POST['message']
            pdf_id = request.session.get('uploaded_pdf_id')
            pdf = UploadedPDF.objects.get(id=pdf_id)
            pdf = pdf.pdf_file.path
            if pdf is not None or pdf_id is not None:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                embeddings = OpenAIEmbeddings()
                knowledge_base = FAISS.from_texts(chunks, embeddings)
                if(len(pdf)>0 ):
                    docs = knowledge_base.similarity_search(message)
                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type="stuff")
                    response = chain.run(input_documents=docs, question=message)
                    return JsonResponse({'status': 'success', 'pdf_id':pdf_id, 'res':response})
                else:
                    response = "Please make sure to upload your file before proceeding."
                    return JsonResponse({'status': 'success', 'pdf_id':None, 'res':response})
        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid form data'})
    else:
        return render(request, 'chatpdf.html', {'form': form})
    
    return render(request, 'chatpdf.html', {'form': form})



def reportGeneretore(request):
    if request.method == 'POST':
        if 'submit' in request.POST:
            client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
            message = request.POST['message']
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[{"role": "system", "content": "You are a helpful report writer."},{"role": "user", "content":message }]
            )
            response = response['choices'][0]['text']
            return JsonResponse({'status': 'success', 'res':response})
    
    return render(request, 'reportgeneratore.html')

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def storyTelling(request):
    if request.method == 'POST':
        if 'submit' in request.POST:
            form = UploadedImageForm(request.POST, request.FILES)
            if form.is_valid():
                image = form.save()
                image = image.image.path
                if image is not None :
                    encoded_image = encode_image(image)
                    openai.api_key = os.environ["OPENAI_API_KEY_A"]
                    result = openai.chat.completions.create(
                        model = "gpt-4-vision-preview",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text",
                                    "text": "Generate a comprehensive and insightful business intelligence narrative for my managers, elucidating the key trends, data points, and strategic insights depicted in the provided image. Craft a compelling story, detailed description, and actionable decision based on the information conveyed by the  image."},
                                    {"type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{encoded_image}"},
                                ]
                            },
                        ],
                        max_tokens=1000
                    )
                    response = result.choices[0].message.content
                    print(textwrap.fill(result.choices[0].message.content, width=70))
            return JsonResponse({'status': 'success', 'res':response})
    
    return render(request, 'storyTelling.html')

def analyzer(request):
    if request.method == 'POST':
        if 'submit' in request.POST:
            form = UploadedImageFile(request.POST, request.FILES)
            if form.is_valid():
                file = form.save()
                file = file.file.path
                df = pd.read_csv(file)
                df = df.sample(n=20, random_state=42)
                text_representation = str(df.to_string(index=False))
                prompt=str("As a data analyst and business intelligence professional, I would like you to provide a detailed analysis of the dataset below. Please offer insights, trends, and any meaningful patterns you observe. Feel free to highlight key metrics, potential correlations, and notable observations. here the data " + text_representation)
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
                prompt = re.sub(r'\s+', ' ', prompt)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to Analyze dataset."},
                        {"role": "user", "content":prompt }
                    ]
                ) 
                response = response.choices[0].message.content
                print(response)
                return JsonResponse({'status': 'success', 'res':response})
    return render(request, 'analyzer.html')


def visualize(request):
    if request.method == 'POST':
        if 'submit' in request.POST:
            form = UploadedImageFile(request.POST, request.FILES)
            if form.is_valid():
                file = form.save()
                df =  pd.read_csv(file.file.path)  
                column_names_text =  ', '.join([f"{col} [{df[col].dtype}]" for col in df.columns])
                example = str(df[:1].to_string(index=False))
                example = re.sub(r'\s+', ' ', example)
                prompt=str("Generate at least 10 suggestions for visualizing insights for power BI based on the given dataset. The dataset includes the following columns(with thier types):" + column_names_text + "\n here is a sample:" + example +"\n start with names of visualizing and then give the description.")
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
                prompt = re.sub(r'\s+', ' ', prompt)
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to Analyze dataset."},
                        {"role": "user", "content":prompt }
                    ]
                ) 
                response = response.choices[0].message.content
                return JsonResponse({'status': 'success', 'res':response})
    return render(request, 'visualize.html')


def color(request):
    if request.method == 'POST':
        if 'message' in request.POST:
            message = request.POST['message']
            if(len(message)>0):
                prompt="Generate 3 suggestions groups(pallets contain more than 5 colors Consistent) of colors for charts and vizualization based on this description: " + message + "\n Please return just the color lists as comma-separated values, each group separated by / no additional text. example: #5E7CE2, #F9A03F, #66C2A5, #8DA0CB, #E78AC3 / #6DB6FF, #FF6768, #5EBA7D, #FFD666, #AD82C1 / #86BBD8, #FF9A6A, #92C0AB, #C795C1, #94D5AB."
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to suggest Consistent colors for visulization."},
                        {"role": "user", "content":prompt }
                    ]
                ) 
                response = response.choices[0].message.content
                palettes = response.split(" / ")
                global_array = [palette.split(", ") for palette in palettes ]
                return JsonResponse({'status': 'success', 'res':response, 'colors': global_array})
    return render(request, 'color.html')

def dax(request):
    if request.method == 'POST':
        if 'message' in request.POST:
            message = request.POST['message']
            if(len(message)>0):
                prompt="Translate the following description into a DAX query: '" + message + "' \n Just the Query no additional text"
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to Turn description into DAX Queries."},
                        {"role": "user", "content":prompt }
                    ]
                ) 
                response = response.choices[0].message.content
                return JsonResponse({'status': 'success', 'res':response})
    return render(request, 'dax.html')



def sql(request):
    if request.method == 'POST':
        if 'message' in request.POST:
            message = request.POST['message']
            if(len(message)>0):
                prompt="Translate the following description into a SQL query: '" + message + "' \n Just the Query no additional text"
                client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY_A"])
                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to Turn description into SQL Queries."},
                        {"role": "user", "content":prompt }
                    ]
                ) 
                response = response.choices[0].message.content
                return JsonResponse({'status': 'success', 'res':response})
    return render(request, 'sql.html')
