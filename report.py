import shelve

def generate_report(title, stats, file):
    f.write(f"----- {title} -----\n")
    if type(stats) == dict:
        for key, value in stats.items():
            f.write(f"{key} {value}\n")
    elif type(stats) == list or type(stats) == set:
        for value in stats:
            f.write(f"{value}\n")
    elif type(stats) == int or type(stats) == str or type(stats) == bool or type(stats) == float:
        f.write(f"{stats}\n")
    f.write("------------------\n")


if __name__ == "__main__":
    with open("report.txt", "w") as f:

        f.write(f"----- Daniel Boghossian (27489674), Owen Koch (86884141)-----\n")

        with shelve.open("indexedDocuments.shelve") as indexedDocuments:
            generate_report("Indexed Documents", indexedDocuments["indexedDocuments"], f)

        with shelve.open("uniqueTokens.shelve") as uniqueTokens:
            generate_report("Unique Tokens", uniqueTokens["uniqueTokens"], f)

        with shelve.open("totalSize.shelve") as totalSize:
            generate_report("Total Size", totalSize["kilobytes"], f)

        with shelve.open("skippedDocuments.shelve") as skippedDocuments:
            generate_report("Skipped Documents", skippedDocuments["skippedDocuments"], f)

        with shelve.open("topTokens.shelve") as topTokens:
            generate_report("Top Tokens", topTokens["topTokens"], f)
    
        