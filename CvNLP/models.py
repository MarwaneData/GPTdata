from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    email = models.EmailField()
    phone_number = models.CharField(max_length=20)
    location = models.CharField(max_length=255)
    description = models.TextField()
    level_education = models.CharField(max_length=255)
    last_company = models.CharField(max_length=255)
    year_of_experience = models.PositiveIntegerField()
    desired_job = models.CharField(max_length=255)
    desired_location = models.CharField(max_length=255)
    skills = models.TextField()

    def __str__(self):
        return f"{self.user.username}'s Profile"
    


class UploadedPDF(models.Model):
    pdf_file = models.FileField(upload_to='uploads/')


class UploadedImage(models.Model):
    image = models.ImageField(upload_to ='uploads/images/') 

class UploadedCsv(models.Model):
    file = models.FileField(upload_to='uploaded_csvs/')