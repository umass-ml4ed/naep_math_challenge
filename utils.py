import csv
from collections import defaultdict
import json
import pandas as pd

# This function adds three infos.
# 1) question text 2) rubric 3) composite score

def add_additional_info(qid, qid_dict):
    qid_dict[qid]["question"] = {}
    qid_dict[qid]["rubric"] = {}
    qid_dict[qid]["comp_score"] = {}

    if qid == "VH134067":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Write a rule that describes the relationship between the input numbers and the output numbers in the table shown."
        # 2) Rubric
        # qid_dict["rubric"]["A"] = {"correct": {"A": 2, "B": 2}, "incorrect": 1}
        qid_dict[qid]["rubric"]["A"] = {"2A": 2, "2B": 2, "1": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = None

    elif qid == "VH266015":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Drag a number into each box to make the following statement true."
        qid_dict[qid]["question"]["B"] = "What must be true about the values of a and b in order for?"
        # 2) Rubric
        qid_dict[qid]["rubric"]["A"] = {"2": 2, "1": 1}
        qid_dict[qid]["rubric"]["B"] = {"4": 4, "3": 3, "2": 2, "1": 1}
        # 3) Order: A_scoreB_score
        qid_dict[qid]["comp_score"] = {"24": 5, "23": 4, "22": 3, "21": 2, "14": 4, "13": 3, "12": 2, "11": 1}

    elif qid == "VH302907":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Use the figure to explain why the sum of the measures of the interior angles of a polygon with 5 sides is 540°"
        qid_dict[qid]["question"]["B"] = "A polygon with 7 side is shown What is the sum of the measures of the interior angles of a polygon with 7 sides?"
        qid_dict[qid]["question"]["C"] = "Based on these two examples, write a possible formula, in terms of n, for the sum of the measures of the interior angles, S, of a polygon with n sides."
        # 2) Rubric
        qid_dict[qid]["rubric"]["A"] = {"2": 2, "1": 1}
        qid_dict[qid]["rubric"]["B"] = {"2": 2, "1": 1}
        qid_dict[qid]["rubric"]["C"] = {"2": 2, "1": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = {"222": 5, "221": 4, "212": 4, "211": 3, "122": 3, "121": 2, "112": 3, "111": 1}

    elif qid == "VH507804":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Farelle has one hundred cards, each labeled with a different number from 1 to 100. Farelle selects four of the cards. How could she place three of the cards in the expression to get the largest result? Drag a card into each box in the expression to show to answer."
        qid_dict[qid]["question"]["B"] = "Next, Farelle selects four new cards. For any four cards, what is a rule about where Farelle should place the new numbers in the same expression to get the largest result?"
        # 2) Rubric
        qid_dict[qid]["rubric"]["A"] = {"2": 2, "1": 1}
        qid_dict[qid]["rubric"]["B"] = {"3": 3, "2A": 2, "2B": 2, "1": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = {"23": 4, "22A": 3, "22B": 3, "21": 2, "13": 3, "12A": 2, "12B": 2, "11": 1}

    elif qid == "VH139380":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "The first six numbers of a number pattern are shown. 5, 8, 11, 14, 17, 20, … The pattern continues by adding the same number each time. What is the next number in the pattern?"
        qid_dict[qid]["question"]["B"] = "Write a rule that can be used to find any number in the pattern, after the first number"
        # 2) Rubric / Only one rubric. 
        qid_dict[qid]["rubric"]["A"] = {"3": 3, "2A": 2, "2B": 2, "1": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = None

    elif qid == "VH266510_2019" or "VH266510_2017":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Which of the following statements must be true about any two distinct lines that intersect at exactly one point in the xy-plane?"
        qid_dict[qid]["question"]["B"] = "Explain how you know."
        # 2) Rubric / Only one rubric. 
        qid_dict[qid]["rubric"]["A"] = {"3": 3, "2A": 2, "2B": 2, "1A": 1, "1B": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = None

    elif qid == "VH269384":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Trent is playing a game. He will pick one card from a bag without looking. The bag holds the following eight cards. If he picks a card with a number greater than 3, he wins the game. What is the probability that Trent will win the game? Trent takes a 5 card out of the bag. Trent puts a 9 card into the bag. Does replacing the card change Trent’s probability of winning the game?"
        qid_dict[qid]["question"]["B"] = "Explain how you know."
        # 2) Rubric
        qid_dict[qid]["rubric"]["A"] = {"2": 2, "1": 1}
        qid_dict[qid]["rubric"]["B"] = {"3": 2, "2A": 2, "2B": 2, "1A": 1, "1B": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = {"11A": 1, "11B": 1, "12A": 3, "12B": 2, "13": 3, "21A": 2, "21B": 2, "22A": 3, "22B": 2, "23": 4}

    elif qid == "VH271613":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "Use the following information to answer questions. Phil’s age is 3 times Alex’s age. Phil is 2 years old than Zach. Tim is 6 years younger than Phil. What is the relationship between Zach’s age and Tim’s age? Use the drop-down menus to show your answer. Zach is (4 or 8) years (younger or older) than Tim. Which of the following statements will NOT be true exactly 10 years from now? A. Phil's age will be 3 times Alex's age. B. Phil will be 2 years older than Zach."
        qid_dict[qid]["question"]["B"] = "Explain why the statement you selected will NOT be true."
        # 2) Rubric
        qid_dict[qid]["rubric"]["A"] = {"2": 2, "1": 1}
        qid_dict[qid]["rubric"]["B"] = {"3": 3, "2A": 2, "2B": 2, "1A": 1, "1B": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = {"11A": 1, "11B": 1, "12A": 2, "12B": 2, "13": 3, "21A": 2, "21B": 2, "22A": 3, "22B": 3, "23": 4}

    elif qid == "VH304954":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "$143 - 48 = \\square$. Mark needs to solve the problem shown. He will solve the problem in two steps. First, Mark subtracts 43 from 143. What does Mark need to do next to complete the problem?"
        qid_dict[qid]["question"]["B"] = "What is the answer to 143 - 48?",
        # 2) Rubric / Only one rubric. 
        qid_dict[qid]["rubric"]["A"] = {"3": 3, "2A": 2, "2B": 2, "1": 1}
        # 3) Composite score
        qid_dict[qid]["comp_score"] = None

    elif qid == "VH525628":
        # 1) Question
        qid_dict[qid]["question"]["A"] = "The letters w, x, y, and z represent four positive integers such that w > x > y > z. How can w, x, y, and z be placed in the following expression to create the least value? Drag a letter into each box to show your answer."
        qid_dict[qid]["question"]["B"] = "Explain how you know that your answer is correct for any such values for w, x, y, and, z."
        # 2) Rubric / Only one rubric. 
        qid_dict[qid]["rubric"]["A"] = {"3": 3, "2A": 2, "2B": 2, "1": 1}  
        # 3) Composite score
        qid_dict[qid]["comp_score"] = None

    else:
        raise Exception("Unrecognized Question Id.")
