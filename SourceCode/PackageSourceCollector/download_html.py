import os, re
import subprocess
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def download(dir_name, size):
    if not os.path.exists(CURRENT_DIR + "/../DataShare/html/" + dir_name):
        os.mkdir(CURRENT_DIR + "/../DataShare/html/" + dir_name)
    else:
        print("WRONG DIR")
        return -1
    for i in range(size):
        cmd = "cd " + CURRENT_DIR + "/../DataShare/html/" + dir_name + ";curl -o ./page_" + str(i + 1) + \
              ".html -G -d 'o=-created' -d 'q=' -d 'c=Programming+Language+%3A%3A+Python+%3A%3A+3' -d 'page=" + str(i + 1) + "' https://pypi.org/search/"
        p = subprocess.Popen(cmd, shell=True)
        try:
            p.wait(timeout=25)
        except subprocess.TimeoutExpired:
            print("kill subprocess !!!!!!!!!!")
            p.kill()

    cmd = "cd " + CURRENT_DIR + "/../DataShare/html/" + dir_name + ";grep -h '<span class=\"package-snippet__name\">' *.html > name.txt"
    os.system(cmd)
    cmd = "rm ./name.txt; cp " + CURRENT_DIR + "/../DataShare/html/" + dir_name + "/name.txt ./name.txt"
    os.system(cmd)

    with open("./name.txt", "rt")as f:
        lines = f.read()
    with open("./name.txt", "wt")as f:
        lines = lines.replace("      <span class=\"package-snippet__name\">", "")
        lines = lines.replace("</span>", "")
        f.write(lines)

    return 0


def download_continue(dir_name, size):
    with open("./name.txt", "r") as last_name_file:
        last_name_lines = last_name_file.readlines()
    last_name_list = []
    for i in last_name_lines:
        last_name_list.append(i.strip())
    if not os.path.exists(CURRENT_DIR + "/../DataShare/html/" + dir_name):
        os.mkdir(CURRENT_DIR + "/../DataShare/html/" + dir_name)
    else:
        print("WRONG DIR")
        return -1
    for i in range(size):
        cmd = "cd " + CURRENT_DIR + "/../DataShare/html/" + dir_name + ";curl -o ./page_" + str(i + 1) + \
              ".html -G -d 'o=-created' -d 'q=' -d 'c=Programming+Language+%3A%3A+Python+%3A%3A+3' -d 'page=" + str(i + 1) + "' https://pypi.org/search/"
        p = subprocess.Popen(cmd, shell=True)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("kill subprocess !!!!!!!!!!")
            p.kill()

        cmd = "cd " + CURRENT_DIR + "/../DataShare/html/" + dir_name + ";grep -h '<span class=\"package-snippet__name\">' *.html > name.txt"
        os.system(cmd)
        cmd = "rm ./name.txt; cp " + CURRENT_DIR + "/../DataShare/html/" + dir_name + "/name.txt ./name.txt"
        os.system(cmd)

        with open("./name.txt", "rt")as f:
            lines = f.read()
            lines = lines.replace("      <span class=\"package-snippet__name\">", "")
            lines = lines.replace("</span>", "")

        with open("./name.txt", "wt")as f:
            f.write(lines)
        with open("./name.txt", "r")as f:
            lines = f.readlines()
        count = 0
        pkg_in_page = 0
        break_sign = 0
        for i in lines:
            name = i.strip()
            if name in last_name_list:
                count += 1
            pkg_in_page += 1
            if count >= 19:
                break_sign = 1
            if pkg_in_page == 20:
                print(count)
                count = 0
                pkg_in_page = 0
        print("########################")

        if break_sign == 1:
            break

    return 0


if __name__ == "__main__":
    download("20230308-1", 6)