from pydantic import BaseModel

class student(BaseModel):
    name:str
    
    
object ={"name":"rishi"}

Student =student(**object)

print(Student)