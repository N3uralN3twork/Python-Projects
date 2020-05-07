import bs4
import requests


def getFormFields(URL):
    res = requests.get(URL)
    res.raise_for_status()
    soup = bs4.BeautifulSoup(res.text, 'html.parser')
    elems = soup.findAll('label')  # Attribute can be changed as per the element to be scraped
    return elems


def printElems(elems):
    file = open('OnlineForm.docx', 'a')
    for i in elems:
        print(i.text.strip())
        file.write(i.text.strip())
        file.write('\n\n')
    file.close()


print('Enter the URL of the page you want to scrape')
URL = input()
# print('Enter the CSS selector you want to find on the page')
# cssSelect = input()

elems = getFormFields(URL)
printElems(elems)