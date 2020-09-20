# whatsapp_chatbot_python
The program will use libraries like nltk and tensorflow to train data on types of user input patterns and uses twilio and flask to deploy the chatbot on whatsapp

The video series will have 4 parts:
	1. data pre-processing and Training 
	2. Creating the Chat-bot 
	3. using Twilio and flask to chat with bot
	4. deploying the bot online


--------------------------------------------------------------------------------------------------------------------------------------------------------
PART - 1

1.Initialization and data pre-processing
Note: Assuming basic knowledge of tensorflow and python programming
	1. Installing and activating virtual environment


		* installing and creating a virtual environment (on windows)
			* py -m pip install --user virtualenv

		* Creating virtual environment
			* py -m venv chbot
               		(env - name of your environment)

	1. Activating virtual environment

		* .\env\Scripts\activate   --> going to scripts directory in env and activating it
	2. Deactivating virtual environment

		* .\env\Scripts\deactivate
	3. installing dependencies like tensorflow, nltk, numpy etc. using "pip install" in the virtual environment
	4. Create an intents.json to hold all intents, user-query and response for our bot to be trained on
	5. data pre-processing using pandas and tokenizer

		* getting tags and (tag, user-query) from intents
		* forming a data frame using of (tag, user-query) to extract training data
		* removing stop words

			* (i, you, is, the, etc.)
		* stemming the words

			* (playing, plays, played) -> play
		* using stemmed words and tags combination to make training sets
		* one-hot encoding tags as we are using softmax activation function
		* Using tokenizer to get text converted into sequence of equal length, getting max length of sequence and vocab size.
	6. training the data on stemmed sentences and intents using tensorflow
	7. saving model, tokenizer and max_sequence length variable

--------------------------------------------------------------------------------------------------------------------------------------------------------
PART - 2

2.Creating the Chatbot 
	* We want to take input from user and use trained model to predict the proper response for it. 


	1. Take and instantiate intents.json, model, max_seq_length and tokenizer from training file
	2. Take input from the user 
	3. Convert the user input into stem form, and then convert to sequence to provide as input for model
	4. Predict the intent(tag) of the user query, and giving out response from intents file
	5. Make it automatic 


--------------------------------------------------------------------------------------------------------------------------------------------------------
PART - 3

3.using Twilio and flask to chat with bot
	1. Creating twilio account to get Whatsapp API
	2. Install flask 
	3. Define an app using flask and inbuilt twilio API and run it 

		* https://www.twilio.com/docs/whatsapp/tutorial/send-and-receive-media-messages-whatsapp-python 
		* It will provide a web application in your local machine
	4. Download "ngrok" to get temporary online web application

		* download ngrok from https://ngrok.com/download 
		* unzip it and install it (usually installed in program files(x86) folder in windows)
		* use command prompt to go to the directory where ngrok is installed and give command

        		ngrok http <port number>
		* It will give temporary online web application which can be provided to twilio to interact with chatbot

--------------------------------------------------------------------------------------------------------------------------------------------------------
PART - 4

4.Deploying the application on heroku
	1. Download and install git

		* https://git-scm.com/downloads 
	2. Signup to Heroku and download Heroku CLI

		* heroku CLI: https://devcenter.heroku.com/articles/heroku-cli 
		* install gunicorn: "pip install gunicorn"

			* will help in deploying app on heroku
			* documentation for gunicorn: https://devcenter.heroku.com/articles/python-gunicorn
	3. Heroku deployment with python help: https://devcenter.heroku.com/articles/getting-started-with-python?singlepage=true
	4. Create helper files

		1. Procfile

			* no extension
			* paste this in procfile: "web: gunicorn chatting:app"

				* chatting - name of the python script (chatting.py)
		2. create requirements.txt 

			* use "pip freeze > requirements.txt" to collect all dependencies required for app to run
		3. runtime.txt

			* specify the version of python used

				* paste "python-3.7.4"
		4. .gitignore

			* contains all the files that are local and are not required by heroku

				* chbot/ (name of the virtual environment)
				* *.pyc
	5. set-up git repository on the directory

			1. initialize repository 
				* git init

			2. add all un-tracked files to repository for commit
				* git add .

			3. commit the changes to repository
				* git commit -m "your commit message"

	6. Create heroku app and push changes through git 

			1. create heroku app 

				* heroku login (for the first time)
				* heroku create <app_name>
				* git push heroku master 




Congratulations !! we have achieved it 








