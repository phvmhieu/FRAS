import yagmail
import os
import datetime

# Get today's date
date = datetime.date.today().strftime("%B %d, %Y")

# Define the path to the directory
path = 'Attendance'

# Change working directory to the folder where the files are stored
os.chdir(path)

# Get the sorted list of files in the directory by modification time
files = sorted(os.listdir(os.getcwd()), key=lambda x: os.path.getmtime(os.path.join(os.getcwd(), x)))

# Get the newest file
newest = files[-1]
filename = os.path.join(os.getcwd(), newest)  # Full file path

# Define the email subject
sub = "Attendance Report for " + str(date)

# Define the body of the email
body = "Please find the attached attendance report for " + str(date) + "."

# Define the receiver email address
receiver = "receiveremail@email.com"

# Mail information (update with correct email and password)
yag = yagmail.SMTP("phamvanhieu06032003@gmail.com", "password")

try:
    # Send the email with attachment
    yag.send(
        to=receiver,
        subject=sub,  # Email subject
        contents=body,  # Email body
        attachments=filename  # File to attach
    )
    print("Email Sent!")
except Exception as e:
    print(f"An error occurred: {e}")
