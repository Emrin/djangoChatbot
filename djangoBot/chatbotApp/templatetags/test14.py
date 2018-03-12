
from django import template
register = template.Library()

def funct(entry):
    return "SOLUTEC : " + entry



register.filter('funct', funct)






