from PackageSourceCollector.download_html import download, download_continue
from PackageSourceCollector.Downloader import download_part_by_txt, download_failure_one, download_json, \
    download_benign_part_by_index, download_part_by_index
from PackageSourceCollector.Unpackage import unpackage_main
# from PackageSourceCollector.Crawler import get_index
import os


def download_new_updated_pkgs(datashare_path, packages_path, date, pages, target_list):
    dir = datashare_path + "/" + packages_path + "/"

    ret = download_continue(date, pages)
    if ret != -1:
        download_part_by_txt(target_list, packages_path, datashare_path)
        download_failure_one(packages_path, datashare_path)
        unpackage_main(datashare_path, packages_path, dir)
        download_json(date, True)


def download_benign_pkgs(datashare_path, packages_path, date, target_list, url, label, attr, index, download_num):
    dir = datashare_path + "/" + packages_path + "/"
    # get_index(url, label, attr, index)

    download_benign_part_by_index(index, packages_path, datashare_path, download_num, date)
    download_failure_one(packages_path, datashare_path)
    # unpackage_main(datashare_path, packages_path, dir)
    download_json(date, False)


def download_random_pkgs(datashare_path, packages_path, date, target_list, url, label, attr, index, download_num):
    dir = datashare_path + "/" + packages_path + "/"
    # get_index(url, label, attr, index)

    download_part_by_index(index, packages_path, datashare_path, download_num, date)
    download_failure_one(packages_path, datashare_path)
    # unpackage_main(datashare_path, packages_path, dir)
    download_json(date, False)


if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datashare_path = os.path.join(PROJECT_DIR, "../Packages/")

    target_list = "./name.txt"

    ################################################
    date = "20230824_1"
    pages = 200
    packages_path = "Packages_0824_1_pypi_tar"
    ################################################

    url = 'https://pypi.org/simple/'
    label = 'a'
    attr = 'href'
    index = "./AllPackagesIndex-pypi.txt"
    download_num = 12000
    # download_random_pkgs(datashare_path, packages_path, date, target_list, url, label, attr, index, download_num)
    download_new_updated_pkgs(datashare_path, packages_path, date, pages, target_list)