#ifndef HPCARVER_PTHREAD_H
#define HPCARVER_PTHREAD_H

namespace hpc_pthread {

/**
 * Initialize pthreads to use thread_count threads.
 * @param thread_count Number of threads.
 */
void hpc_pthread_init(int thread_count);

} // carver

#endif //HPCARVER_PTHREAD_H
