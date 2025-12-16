from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Person(BaseModel):
    name: str = "Noor"
    age: int 
    email: EmailStr
    cgpa: float = Field(default=5, gt=0,lt=10, description="CGPA of a person")
    gender:Optional[str] = None



personA_dict={"name":"Nashid", "age":26,"email":"nashid@gmail.com","gender":"M"}

personA= Person(**personA_dict)
print(personA)
print(personA.name)
print(personA.age)
print(type(personA))

personB= Person(**{'age':50,"email":'noor@gmail.com'})  # we can also write this like personB= Person(age:50)
print(personB)
print(personB.gender)


personB_fromPydantic_to_dict=personB.model_dump()
print(type(personB_fromPydantic_to_dict))
print(personB_fromPydantic_to_dict['age'])

personB_fromPydantic_to_JSON= personB.model_dump_json()
print(type(personB_fromPydantic_to_JSON))