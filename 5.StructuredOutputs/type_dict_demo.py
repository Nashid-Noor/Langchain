from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

personA: Person = {'name':'Nashid','age':26}

print(personA)
print(type(personA))
print(personA['name'])