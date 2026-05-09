from setuptools import setup, find_packages

setup(
    name='999-cli',
    version='0.1.0',
    packages=find_packages(),
    py_modules=['main'],
    install_requires=[
        'langgraph',
        'openai',
        'rich',
        'python-dotenv',
        'pydantic',
        'fastapi',
        'uvicorn',
        'beautifulsoup4',
        'faiss-cpu',
        'sentence-transformers'
    ],
    entry_points={
        'console_scripts': [
            '999=main:main',
        ],
    },
)
