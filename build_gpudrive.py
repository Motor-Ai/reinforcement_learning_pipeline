import subprocess
import os
import logging
import shutil

logging.basicConfig(level=logging.INFO)


def main():
    # Cloning the repository, although typically you would not do this in the build step
    # as the code should already be present. Including it just for completeness.

    # Create and enter the build directory
    if os.path.exists("build"):
        shutil.rmtree("build")
    os.mkdir("build")
    os.chdir("build")

    # Run CMake and Make
    subprocess.check_call(
        [
            "cmake", 
            "../src/envs/gpudrive",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ])
    subprocess.check_call(["find", "external", "-type", "f", "-name", "*.tar", "-delete"])
    subprocess.check_call(["make", f"-j{os.cpu_count()}"])  # Utilize all available cores

    # Going back to the root directory
    os.chdir("..")


if __name__ == "__main__":
    logging.info("Building the C++ code and installing the Python package")
    main()
