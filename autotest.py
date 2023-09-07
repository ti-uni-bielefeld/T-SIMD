#!/usr/bin/env python3

import multiprocessing
import os
import random
import time
import sys

SDE_PATH = ""

COMPILERS = [
    "clang++",
    "g++",
]

SDE_OPTIONS = "-align_checker_action ignore -future"

LOG_DIR = f"autotest_{os.popen('hostname').read().strip()}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
TMP_BUILD_DIR = f"/tmp/T-SIMD_autotest_{time.strftime('%Y-%m-%d_%H-%M-%S')}"


def is_feature_supported(supported_cpu_features, feature):
    feature = feature.strip().lower().replace("_", "").replace("-", "").replace(".", "")
    for supported_cpu_feature in supported_cpu_features:
        if supported_cpu_feature == feature:
            return True
        if supported_cpu_feature.startswith(feature):
            return True
        if supported_cpu_feature.endswith(feature):
            return True

    return False

def get_supported_cpu_features():
    supported_cpu_features = []
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if line.startswith("flags"):
                supported_cpu_features = line.split(": ")[1].split(" ")
                break

    for i in range(len(supported_cpu_features)):
        supported_cpu_features[i] = supported_cpu_features[i].strip().lower().replace("_", "").replace("-", "").replace(".", "")

    return supported_cpu_features

def get_required_emulator(arch_flags):
    SDE_EMULATOR = f"{SDE_PATH} {SDE_OPTIONS} -- "

    supported_cpu_features = get_supported_cpu_features()

    req_cpu_features = []
    for arch_flag in arch_flags.split(" "):
        if arch_flag.startswith("-m"):
            req_cpu_features.append(arch_flag[2:])

    if all([is_feature_supported(supported_cpu_features, req_cpu_feature) for req_cpu_feature in req_cpu_features]):
        # all required cpu features are supported
        # so we don't need to use the emulator
        return ""
    else:
        # there is at least one required cpu feature that is not supported
        # so we need to use the emulator
        return SDE_EMULATOR


def generate_test_configs():
    test_configs = []

    for compiler in COMPILERS:
        if os.system(f"which {compiler} > /dev/null 2>&1") != 0:
            print(f"WARNING: compiler \"{compiler}\" not found, skipping")
            continue

        test_configs.append({
            "compiler": compiler,
            "opt_flags": "-O0",
            "arch_flags": "",
            "sandbox_defines": "-DSIMDVEC_SANDBOX",
            "emulator": "",
        })

        for opt_flags in [
            # "-O0", compiling with -O0 takes an unreasonable amount ram (like up to 70 GB (wtf?))
            #"-O1",
            "-O2",
            #"-O3",
            "-O3 -funroll-loops",
        ]:
            for arch_flags in [
                "-msse2",
                "-msse3",
                "-mssse3",
                "-msse4",
                "-msse4.1",
                "-msse4.2",
                "-mavx",
                "-mavx2",
                "-mavx512f",
                "-mavx512bw",
                "-mavx512dq",
                "-mavx512vbmi",
                "-mavx512bw -mavx512dq",
                "-mavx512dq -mavx512vbmi",
                # TODO: maybe add more combinations?
            ]:
                test_configs.append({
                    "compiler": compiler,
                    "opt_flags": opt_flags,
                    "arch_flags": arch_flags,
                    "sandbox_defines": "",
                    "emulator": get_required_emulator(arch_flags),
                })

    return test_configs


def run_test(test_config):
    print(f"Running test: {test_config}")
    name = f"{test_config['compiler']}_{test_config['opt_flags']}_{test_config['arch_flags']}_{test_config['sandbox_defines']}"
    name = name.replace(" ", "")

    build_dir = f"{TMP_BUILD_DIR}/{name}"

    # make directory if it doesn't exist
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    build_command = f"make build_dir=\"{build_dir}\" compiler=\"{test_config['compiler']}\" optflags=\"{test_config['opt_flags']}\" flags_arch=\"{test_config['arch_flags']}\" sandbox_defines=\"{test_config['sandbox_defines']}\" all_tsimd autotest"

    log_file_compile = f"{LOG_DIR}/{name}_compile.log"
    os.system(
        f"make build_dir=\"{build_dir}\" clean > {log_file_compile} 2>&1")

    os.system(f"{build_command} >> {log_file_compile} 2>&1")

    if test_config["sandbox_defines"] == "":
        log_file_test0 = f"{LOG_DIR}/{name}_test0.log"
        os.system(
            f"{test_config['emulator']} {build_dir}/simdvecautotest0 \"\" 10000 > {log_file_test0} 2>&1 || echo \"ERROR: simdvecautotest0 failed\" >> {log_file_test0}")
        log_file_test1 = f"{LOG_DIR}/{name}_test1.log"
        os.system(
            f"{test_config['emulator']} {build_dir}/simdvecautotest1 \"\" 10000 > {log_file_test1} 2>&1 || echo \"ERROR: simdvecautotest1 failed\" >> {log_file_test1}")
        log_file_testM = f"{LOG_DIR}/{name}_testM.log"
        os.system(
            f"{test_config['emulator']} {build_dir}/simdvecautotestM \"\" 10000 > {log_file_testM} 2>&1 || echo \"ERROR: simdvecautotestM failed\" >> {log_file_testM}")
        log_file_test_load_store = f"{LOG_DIR}/{name}_test_load_store.log"
        os.system(
            f"{test_config['emulator']} {build_dir}/simdvecautotest_load_store 10000 > {log_file_test_load_store} 2>&1 || echo \"ERROR: simdvecautotest_load_store failed\" >> {log_file_test_load_store}")

    os.system(f"rm -r {build_dir}")

    print(f"Finished executing test: {test_config}")


if __name__ == "__main__":
    is_sde_path_in_env = False
    if "SDE_PATH" in os.environ:
        SDE_PATH = os.environ["SDE_PATH"]
        is_sde_path_in_env = True

    is_sde_path_in_args = False
    for i in range(len(sys.argv)):
        if sys.argv[i] == "--sde-path":
            SDE_PATH = sys.argv[i + 1]
            is_sde_path_in_args = True
        elif sys.argv[i].startswith("SDE_PATH="):
            SDE_PATH = sys.argv[i].split("=")[1]
            is_sde_path_in_args = True

    test_configs = generate_test_configs()

    is_sde_required = any([test_config["emulator"] != "" for test_config in test_configs])

    if is_sde_required and  not os.path.exists(SDE_PATH):
        if is_sde_path_in_args:
            print(
                f"ERROR: Intel® SDE is required for some tests, but SDE_PATH from command line arguments is invalid: \"{SDE_PATH}\"")
        elif is_sde_path_in_env:
            print(
                f"ERROR: Intel® SDE is required for some tests, but SDE_PATH from environment variables is invalid: \"{SDE_PATH}\"")
        else:
            print(f"ERROR: Intel® SDE is required for some tests, but SDE_PATH from constant is invalid: \"{SDE_PATH}\"")
        print("       Please specify a valid SDE_PATH pointing to the sde or sde64 executable (download from https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) with one of the following options:")
        print("       - with the --sde-path command line argument")
        print("       - with the SDE_PATH= command line argument")
        print("       - with the SDE_PATH environment variable")
        print("       - by changing the SDE_PATH constant in this script")
        sys.exit(1)

    os.nice(19)  # be as nice as possible
    num_procs_cores = multiprocessing.cpu_count()
    ram_size_kb = 0
    with open("/proc/meminfo", "r") as f:
        for line in f.readlines():
            if line.startswith("MemTotal"):
                ram_size_kb = int(line.split()[1])
                break

    # reserve 15 GB per process to be sure
    num_procs_ram = ram_size_kb // (15 * 1024 * 1024)

    num_procs = min(num_procs_cores, num_procs_ram)

    print(f"Using {num_procs} processes ({num_procs_cores} based on number of cores, {num_procs_ram} based on RAM size)")

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(TMP_BUILD_DIR):
        os.makedirs(TMP_BUILD_DIR)

    pool = multiprocessing.Pool(processes=num_procs)

    random.shuffle(test_configs)

    print(f"Executing {len(test_configs)} tests...")

    start_time = time.time()

    for test in test_configs:
        pool.apply_async(run_test, args=(test,))

    pool.close()
    pool.join()

    os.system(
        f"grep -nri --exclude=errors.log --exclude=warnings.log \"error\" {LOG_DIR} > {LOG_DIR}/errors.log")
    os.system(
        f"grep -nri --exclude=errors.log --exclude=warnings.log \"warning\" {LOG_DIR} > {LOG_DIR}/warnings.log")

    os.system(
        f"echo \"Number of errors (see file {LOG_DIR}/errors.log):\" && cat {LOG_DIR}/errors.log | wc -l")
    os.system(
        f"echo \"Number of warnings (see file {LOG_DIR}/warnings.log): \" && cat {LOG_DIR}/warnings.log | wc -l")

    print(
        f"Done. Executed {len(test_configs)} tests in {time.time() - start_time} seconds")
