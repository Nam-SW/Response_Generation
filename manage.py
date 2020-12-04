import sys
import zipfile
from os import listdir
from os.path import abspath, dirname, join, basename, splitext

argv = sys.argv
root = dirname(abspath(__file__))

if argv[1] == "pack":
    output_file = zipfile.ZipFile(argv[2], "w")

    with open(".datafiles", "r") as f:
        for line in f.readlines():
            line = line.strip()
            fname, extension = splitext(basename(line))

            if fname == "*":
                dir_name = dirname(line)

                for f in listdir(join(root, dir_name)):
                    if f.endswith(extension):
                        output_file.write(
                            join(root, dir_name, f),
                            join(dir_name, f),
                            zipfile.ZIP_DEFLATED,
                        )

            else:
                output_file.write(join(root, line), line, zipfile.ZIP_DEFLATED)

    output_file.close()
elif argv[1] == "unpack":
    zipfile.ZipFile(argv[2], "r").extractall(root)
