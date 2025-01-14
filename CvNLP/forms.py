# forms.py
from django import forms
from .models import UploadedPDF, UploadedImage, UploadedCsv

class PDFUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedPDF
        fields = ['pdf_file']


class UploadedImageForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']

class UploadedImageFile(forms.ModelForm):
    class Meta:
        model = UploadedCsv
        fields = ['file']