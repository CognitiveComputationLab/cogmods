def if_elimination(card, p):
    if card == p:
        return True


def if_elimination_converse(card, q):
    if card == q:
        return True


def not_introduction(card, q):
    q[1] = not q[1]
    if if_elimination(card, q):
        return True


def main():
    p = ["vowel", True]  # vowel
    q = ["even", True]  # even
    cards = {1: ["vowel", True], 2: ["vowel", False], 3: ["even", True], 4: ["even", False]}
    count = 0
    for card in cards.values():
        count += 1
        if if_elimination(card, p) or if_elimination_converse(card, q) or not_introduction(card, q):
            print("Select card: ", count)


if __name__ == "__main__":
    main()
