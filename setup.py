from setuptools import setup, find_packages
from typing import List
HYPEN_E_DOT='-e .'

def get_requirements(file:str)->List[str]:
    requirements=[]
    with open(file,'r') as file:
        content=file.readlines()
        for library in content:
            requirements.append(library.replace("\n",""))
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="student_performance_indicator_app",
    version="0.1.0",
    author="Rajib",
    author_email='jaiswalrajib98192@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
