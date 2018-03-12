from django.shortcuts import render

# Create your views here.


from django.http import HttpResponse


from .templatetags.test14 import funct



def index(request):
    return render(request, 'chatbotApp/index.html', { 'title': 'something1'})


from .templatetags.test14 import funct
from .templatetags.testBot import respond

# def request_page(request):
#     if(request.GET.get('mybtn')):
#         
# from( str(request.GET.get('mytextbox')) )
#     return render(request, 'chatbotApp/index.html', { 'title': 'THING!'})


def req1(request):
    return render(request, 'chatbotApp/hidden.html', { 'text': 'conversation'})

# def ask(request):
#     return render(request, 'chatbotApp/index.html',)


# in layouts{{ "one man's trash is another man's treasure"|funct }}
