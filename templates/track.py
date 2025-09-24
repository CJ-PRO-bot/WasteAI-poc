import phonenumbers
from phonenumbers import geocoder

# Parse the phone number (replace with the number you want to look up)
phone_number = phonenumbers.parse("+97577442271")  # Example number

# Get the location description in English
location = geocoder.description_for_number(phone_number, "en")

# Print the static location information
print(f"The location of the phone number is: {location}")