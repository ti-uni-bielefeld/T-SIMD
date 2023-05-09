import multiprocessing
import os
import random
import time

test_sets = []

for compiler in [
    "clang++",
    "g++",
]:
    for opt_flags in [
        "-O0",
        "-O1",
        "-O2",
        "-O3",
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
            "-mavx512er",
            "-mavx512vbmi",
            "-mavx512bw -mavx512dq",
            "-mavx512bw -mavx512er",
            "-mavx512dq -mavx512er",
            "-mavx512bw -mavx512dq -mavx512er",
            "-mavx512dq -mavx512er -mavx512vbmi",
        ]:
            test_sets.append({
                "compiler": compiler,
                "opt_flags": opt_flags,
                "arch_flags": arch_flags,
            })


def run_test(test_set):
    print("Running test set: {}".format(test_set))
    name = "{}_{}_{}".format(
        test_set["compiler"],
        test_set["opt_flags"],
        test_set["arch_flags"],
    ).replace(" ", "")

    build_dir = "/tmp/T-SIMD_autotest/{}".format(name)

    # make directory if it doesn't exist
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    build_command = "make build_dir=\"{}\" compiler=\"{}\" optflags=\"{}\" flags_arch=\"{}\" all_tsimd autotest".format(
        build_dir,
        test_set["compiler"],
        test_set["opt_flags"],
        test_set["arch_flags"],
    )

    log_file_compile = "autotest/{}_compile.log".format(name)
    os.system("make build_dir=\"{}\" clean > {} 2>&1".format(
        build_dir,
        log_file_compile,
    ))

    os.system("{} >> {} 2>&1".format(build_command, log_file_compile))

    log_file_test0 = "autotest/{}_test0.log".format(name)
    os.system("{}/simdvecautotest0 > {} 2>&1 || echo \"ERROR: simdvecautotest0 failed\" >> {}".format(
        build_dir, log_file_test0, log_file_test0, log_file_test0,
    ))
    log_file_test1 = "autotest/{}_test1.log".format(name)
    os.system("{}/simdvecautotest1 > {} 2>&1 || echo \"ERROR: simdvecautotest1 failed\" >> {}".format(
        build_dir, log_file_test1, log_file_test1, log_file_test1,
    ))
    log_file_testM = "autotest/{}_testM.log".format(name)
    os.system("{}/simdvecautotestM > {} 2>&1 || echo \"ERROR: simdvecautotestM failed\" >> {}".format(
        build_dir, log_file_testM, log_file_testM, log_file_testM,
    ))

    os.system("rm -r {}".format(build_dir))

    print("Finished test set: {}".format(test_set))


if __name__ == "__main__":
    os.nice(19)
    num_procs_cores = multiprocessing.cpu_count()
    num_procs_ram = 0
    with open("/proc/meminfo", "r") as f:
        for line in f.readlines():
            if line.startswith("MemTotal"):
                ram_size_kb = int(line.split()[1])
                # 10 GB per process
                num_procs_ram = ram_size_kb // (10 * 1024 * 1024)
                break

    num_procs = min(num_procs_cores, num_procs_ram)

    # make autotest directory if it doesn't exist
    if not os.path.exists("autotest"):
        os.makedirs("autotest")

    pool = multiprocessing.Pool(processes=num_procs)

    random.shuffle(test_sets)

    print("Executing {} test sets...".format(len(test_sets)))

    start_time = time.time()

    for test_set in test_sets:
        pool.apply_async(run_test, args=(test_set,))

    pool.close()
    pool.join()

    os.system(
        "grep -nri --exclude=errors.log --exclude=warnings.log \"error\" autotest > autotest/errors.log")
    os.system(
        "grep -nri --exclude=errors.log --exclude=warnings.log \"warning\" autotest > autotest/warnings.log")

    os.system(
        "echo \"Number of errors (see file autotest/errors.log):\" && cat autotest/errors.log | wc -l")
    os.system("echo \"Number of warnings (see file autotest/warnings.log): \" && cat autotest/warnings.log | wc -l")

    print("Done. Executed {} test sets in {} seconds".format(
        len(test_sets),
        time.time() - start_time,
    ))
