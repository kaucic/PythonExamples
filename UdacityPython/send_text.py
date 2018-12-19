# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:52:01 2016

@author: 200005550
"""

import twilio
from twilio.rest import TwilioRestClient
 
# twilio phone number 15183175037
 
# Your Account Sid and Auth Token from twilio.com/user/account
account_sid = "AC62105826eedbd59e7e5a83c398d8d3fe"
auth_token  = "3b6fede3cb79eed8eb27b08bddb7ec66"
client = TwilioRestClient(account_sid, auth_token)
 
message = client.messages.create(
    body="Hey Robert, this is dad. " +
    "I can now send you text messages from a Python script using my twilio account." +
    "I havent figured out how to see what you text if you respond to this.",
    to="+15187636809",    # Replace with your phone number
    #to="+15183139263",    # Replace with your phone number
    from_="+15183175037") # Replace with your Twilio number
print message.sid