"""Volume 3: Web Scraping.
Benj McMulin
Math 405
1/16/2024
"""

import re
import requests
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
import os.path

# Problem 1
def prob1():
    """Use the requests library to get the HTML source for the website 
    http://www.example.com.
    Save the source as a file called example.html.
    If the file already exists, do not scrape the website or overwrite the file.
    """
    # Check if file exists
    if not os.path.isfile('example.html'):
        # Get the html source
        html = requests.get('http://www.example.com').text
        
        # Save the source as a file
        with open('example.html', 'w') as file:
            file.write(html)
    return None
    
# Problem 2
def prob2(code):
    """Return a list of the names of the tags in the given HTML code.
    Parameters:
        code (str): A string of html code
    Returns:
        (list): Names of all tags in the given code"""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(code, 'html.parser')
    
    # Get all the tags
    tags = soup.find_all(True)
    
    # Get the names of the tags
    tag_names = [tag.name for tag in tags]
    
    return tag_names


# Problem 3
def prob3(filename="example.html"):
    """Read the specified file and load it into BeautifulSoup. Return the
    text of the first <a> tag and whether or not it has an href
    attribute.
    Parameters:
        filename (str): Filename to open
    Returns:
        (str): text of first <a> tag
        (bool): whether or not the tag has an 'href' attribute
    """
    # Open the file and parse it with BeautifulSoup
    with open(filename, 'r') as file:
        data = file.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Get the first <a> tag
    a_tag = soup.find('a')

    # Check if it has an href attribute
    has_href = hasattr(a_tag, 'href')

    return a_tag.text, has_href


# Problem 4
def prob4(filename="san_diego_weather.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the following tags:

    1. The tag containing the date 'Thursday, January 1, 2015'.
    2. The tags which contain the links 'Previous Day' and 'Next Day'.
    3. The tag which contains the number associated with the Actual Max
        Temperature.

    Returns:
        (list) A list of bs4.element.Tag objects (NOT text).
    """
    # Open the file and parse it with BeautifulSoup
    with open(filename, 'r') as file:
        data = file.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Get the tags containing the specified text
    first = soup.find(string="Thursday, January 1, 2015").parent
    second = soup.find_all(class_="previous-link")[0]
    third = soup.find_all(class_="next-link")[0]
    fourth = soup.find(string="Max Temperature").parent.parent.next_sibling.next_sibling.span.span

    return [first, second, third, fourth]

# Problem 5
def prob5(filename="large_banks_index.html"):
    """Read the specified file and load it into BeautifulSoup. Return a list
    of the tags containing the links to bank data from September 30, 2003 to
    December 31, 2014, where the dates are in reverse chronological order.

    Returns:
        (list): A list of bs4.element.Tag objects (NOT text).
    """
    # Open the file and parse it with BeautifulSoup
    with open(filename, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Get all the tags
    my_tags = soup.find_all(name='a')

    # Get the first and last tags
    first_tag = soup.find(string="December 31, 2014").parent
    last_tag = soup.find(string="September 30, 2003").parent

    # Get the tags between the first and last tags
    append = False
    list_of_tags = []

    for tag in my_tags:

        # Check if the tag is the first tag
        if tag == first_tag:
            append = True

        # Check if the tag is the last tag
        if append:

            # Check if the tag has an href attribute
            if hasattr(tag, 'href'):
                list_of_tags.append(tag)

        # Check if the tag is the last tag
        if tag == last_tag:
            break

    return list_of_tags


# Problem 6
def prob6(filename="large_banks_data.html"):
    """Read the specified file and load it into BeautifulSoup. Create a single
    figure with two subplots:

    1. A sorted bar chart of the seven banks with the most domestic branches.
    2. A sorted bar chart of the seven banks with the most foreign branches.

    In the case of a tie, sort the banks alphabetically by name.
    """
    # Open the file and parse it with BeautifulSoup
    with open(filename, 'r') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'html.parser')

    # Get the tags containing the data
    bank_tags = [tag.td.text[:tag.td.text.find("/")] for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]
    bank_ids = [tag.td.next_sibling.next_sibling.next_sibling.next_sibling.text
                for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]
    domestic_branches = [tag.td.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text
                         for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]
    foreign_branches = [tag.td.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text
                         for tag in soup.find_all(name='tr', attrs={"valign": "TOP"})][1:]

    # Replace "." with "0"
    for i, bank in enumerate(domestic_branches):
        if bank == ".":
            domestic_branches[i] = "0"
            foreign_branches[i] = "0"

    # Convert to floats
    domestic_banks = {(bank, bank_ids[i]): float(domestic_branches[i].replace(",", "")) for i, bank in enumerate(bank_tags)}
    foreign_banks = {(bank, bank_ids[i]): float(foreign_branches[i].replace(",", "")) for i, bank in enumerate(bank_tags)}

    # Sort the dictionaries
    sorted_domestic = sorted(domestic_banks.items(), key=lambda item: item[1], reverse=True)
    sorted_foreign = sorted(foreign_banks.items(), key=lambda item: item[1], reverse=True)

    # First Plot
    fig, axes = plt.subplots(nrows = 2, ncols = 1)
    axes[0].barh([x[0][0] for x in sorted_domestic[:7]], [x[1] for x in sorted_domestic[:7]])

    # Second Plot
    axes[1].barh([x[0][0] for x in sorted_foreign[:7]], [x[1] for x in sorted_foreign[:7]])
    plt.title("Banks with Most Foreign Branches")
    plt.suptitle("Banks with Most Domestic Branches")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Uncomment the following lines to test each problem
    # print(prob1())

    # TODO: Find way to test prob2
    # result = prob2("example.html")
    # print(result)
    
    # print(prob3())

    # print(prob4())

    # print(prob5())
    
    # prob6()
    pass