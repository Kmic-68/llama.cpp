#define _GNU_SOURCE
#include <signal.h>
#include <time.h>
#include <execinfo.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <sys/syscall.h>

#define SIGSAMP (SIGRTMIN + 5)
#define MAXS 400000
#define DEPTH 6
static void *samples[MAXS][DEPTH];
static volatile int nsamp = 0;
static volatile int stop = 0;

static void handler(int sig, siginfo_t * si, void * uc) {
    (void) sig; (void) si; (void) uc;
    void *bt[DEPTH + 3];
    int n = backtrace(bt, DEPTH + 3);
    int i = nsamp;
    if (i < MAXS) {
        for (int k = 0; k < DEPTH; k++) samples[i][k] = (n > k + 2) ? bt[k + 2] : NULL;
        nsamp = i + 1;
    }
}

static unsigned long thread_cpu(int tid) {
    char p[64]; snprintf(p, sizeof p, "/proc/self/task/%d/stat", tid);
    FILE * f = fopen(p, "r"); if (!f) return 0;
    char buf[1024]; size_t n = fread(buf, 1, sizeof buf - 1, f); fclose(f);
    buf[n] = 0;
    char * c = strrchr(buf, ')'); if (!c) return 0;
    unsigned long u = 0, s = 0; int fld = 3;
    char * tok = strtok(c + 2, " ");
    while (tok) { if (fld == 14) u = strtoul(tok, 0, 10); if (fld == 15) { s = strtoul(tok, 0, 10); break; } fld++; tok = strtok(0, " "); }
    return u + s;
}

static void * sampler(void * arg) {
    (void) arg;
    sigset_t m; sigfillset(&m); pthread_sigmask(SIG_BLOCK, &m, NULL); // never sample the sampler
    int self = (int) syscall(SYS_gettid);
    int busiest = 0; unsigned long prev[512]; int tids[512]; int ntid = 0;
    memset(prev, 0, sizeof prev);
    int rescan = 0;
    while (!stop) {
        if (rescan-- <= 0) { // every ~250 ms find the busiest thread
            DIR * d = opendir("/proc/self/task"); ntid = 0;
            struct dirent * e;
            while (d && (e = readdir(d)) && ntid < 512) { int t = atoi(e->d_name); if (t > 0 && t != self) tids[ntid++] = t; }
            if (d) closedir(d);
            unsigned long best = 0;
            for (int i = 0; i < ntid; i++) { unsigned long c = thread_cpu(tids[i]); unsigned long d2 = c - prev[i]; prev[i] = c; if (d2 >= best) { best = d2; busiest = tids[i]; } }
            rescan = 125;
        }
        if (busiest) syscall(SYS_tgkill, getpid(), busiest, SIGSAMP);
        struct timespec ts = {0, 2000000};
        nanosleep(&ts, NULL);
    }
    return NULL;
}

__attribute__((constructor)) static void prof_init(void) {
    struct sigaction sa; memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = handler; sa.sa_flags = SA_RESTART | SA_SIGINFO; sigemptyset(&sa.sa_mask);
    sigaction(SIGSAMP, &sa, NULL);
    void * dummy[4]; backtrace(dummy, 4);
    pthread_t th; pthread_create(&th, NULL, sampler, NULL);
}

__attribute__((destructor)) static void prof_fini(void) {
    stop = 1;
    const char * out = getenv("PROF_OUT"); if (!out) out = "/tmp/prof.txt";
    char path[512]; snprintf(path, sizeof path, "%s.%d", out, (int) getpid());
    FILE * f = fopen(path, "w"); if (!f) return;
    int n = nsamp;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < DEPTH; k++) {
            Dl_info di; const char * nm = "?"; const char * lib = "?";
            if (samples[i][k] && dladdr(samples[i][k], &di)) {
                if (di.dli_sname) nm = di.dli_sname;
                if (di.dli_fname) { const char * b = strrchr(di.dli_fname, '/'); lib = b ? b + 1 : di.dli_fname; }
            }
            fprintf(f, "%s%s@%s", k ? ";" : "", nm, lib);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "# total samples %d\n", n);
    fclose(f);
    fprintf(stderr, "prof: wrote %d samples to %s\n", n, path);
}
