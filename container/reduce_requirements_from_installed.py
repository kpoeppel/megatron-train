import sys


def main():
    constraints_file = sys.argv[1]
    requirements_file = sys.argv[2]

    with open(constraints_file, "r") as fp:
        constraints = fp.readlines()
        constraints = {
            constr.split(" @ ")[0].split(">=")[0].split("==")[0].split("!=")[0].split("<=")[0].strip("\n"): constr
            for constr in constraints
        }

    with open(requirements_file, "r") as fp:
        requirements = fp.readlines()
        requirements = {
            constr.split(" @ ")[0].split(">=")[0].split("==")[0].split("!=")[0].split("<=")[0].strip("\n"): constr
            for constr in requirements
        }

    for package in constraints:
        if package in requirements:
            if "@" in constraints[package]:
                del requirements[package]

    for req in requirements.values():
        print(req.strip("\n"))


if __name__ == "__main__":
    main()
