from django import template
register = template.Library()

def respond(entry):
    return "FROM TESTBOT SOLUTEC : " + entry



register.filter('respond', respond)