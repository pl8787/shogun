#ifndef _KERNEL_H___
#define _KERNEL_H___

#include "lib/common.h"
#include "features/Features.h"

#include <stdio.h>

class CKernel
{
	public:
		CKernel(KERNELCACHE_IDX size);
		virtual ~CKernel();

		/** get kernel function for lhs feature vector x 
		  and rhs feature vector y
		 */
		REAL kernel(INT x, INT y);

		/** initialize kernel cache
		 *  make sure to check that your kernel can deal with the
		 *  supplied features (!)
		 *  set do_init to true if you want the kernel to call its setup function (like getting scaling parameters,...)
		 */
		virtual bool init(CFeatures* lhs, CFeatures* rhs, bool do_init);

		/// clean up your kernel
		virtual void cleanup()=0;

		/// load and save the kernel matrix
		bool load(CHAR* fname);
		bool save(CHAR* fname);

		/// load and save kernel init_data
		virtual bool load_init(FILE* src)=0;
		virtual bool save_init(FILE* dest)=0;
		
		/// get left/right hand side of features used in kernel
		inline CFeatures* get_lhs() { return lhs; } ;
		inline CFeatures* get_rhs() { return rhs;  } ;

		inline void cache_reset() {	resize_kernel_cache(cache_size) ; } ;
		
		/// takes all necessary steps if the lhs is removed from kernel
		virtual void remove_lhs() { if (lhs) cache_reset() ; lhs = NULL ;  } ;
		/// takes all necessary steps if the rhs is removed from kernel
		virtual void remove_rhs() { if (rhs) cache_reset() ; rhs = NULL ;  } ;
		
		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EKernelType get_kernel_type()=0 ;

		/** return feature type the kernel can deal with
		  */
		virtual EFeatureType get_feature_type()=0;

		/** return feature class the kernel can deal with
		  */
		virtual EFeatureClass get_feature_class()=0;

		// return the name of a kernel
		virtual const CHAR* get_name()=0 ;

		// return the size of the kernel cache
		int get_cache_size() { return cache_size; }

		void get_kernel_row(KERNELCACHE_IDX docnum, LONG *active2dnum, REAL *buffer) ;
		void cache_kernel_row(KERNELCACHE_IDX x);
		void cache_multiple_kernel_rows(LONG* key, INT varnum);
		void kernel_cache_reset_lru();

		void resize_kernel_cache(KERNELCACHE_IDX size) ;
		
		/// set the time used for lru	
		inline void set_time(LONG t)
		{
			kernel_cache.time=t;
		}

		// Update lru time to avoid removal from cache.
		KERNELCACHE_IDX kernel_cache_touch(KERNELCACHE_IDX cacheidx)
		{
			if(kernel_cache.index[cacheidx] != -1)
			{
				kernel_cache.lru[kernel_cache.index[cacheidx]]=kernel_cache.time; 
				return(1);
			}
			return(0);
		}

		void list_kernel();

		/** for optimizable kernels, i.e. kernels where the weight 
		 * vector can be computed explicitely (if it fits into memory) 
		 */

		inline double get_combined_kernel_weight() { return combined_kernel_weight; }
		inline void set_combined_kernel_weight(double nw) { combined_kernel_weight=nw; }

		inline bool get_is_initialized() { return optimization_initialized; }

		bool is_optimizable();
		virtual bool init_optimization(INT count, INT *IDX, REAL * weights); 
		virtual bool delete_optimization();
		virtual REAL compute_optimized(INT idx);

	protected:
		inline void set_is_initialized(bool init) { optimization_initialized=init; }

		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual REAL compute(INT x, INT y)=0;

		/**@ cache kernel evalutations to improve speed
		 */
		//@{
		void   kernel_cache_shrink(KERNELCACHE_IDX, KERNELCACHE_IDX, KERNELCACHE_IDX *);

		/// init kernel cache of size megabytes
		void   kernel_cache_init(KERNELCACHE_IDX size);
		void   kernel_cache_cleanup();
		KERNELCACHE_IDX   kernel_cache_malloc();
		void   kernel_cache_free(KERNELCACHE_IDX cacheidx);
		KERNELCACHE_IDX   kernel_cache_free_lru();
		KERNELCACHE_ELEM *kernel_cache_clean_and_malloc(KERNELCACHE_IDX);


		/// Is that row cached?
		inline KERNELCACHE_IDX kernel_cache_check(KERNELCACHE_IDX cacheidx)
		{
			return(kernel_cache.index[cacheidx] != -1);
		}
		//@}

		struct KERNEL_CACHE {
			KERNELCACHE_IDX   *index;  
			KERNELCACHE_ELEM  *buffer; 
			KERNELCACHE_IDX   *invindex;
			KERNELCACHE_IDX   *active2totdoc;
			KERNELCACHE_IDX   *totdoc2active;
			KERNELCACHE_IDX   *lru;
			KERNELCACHE_IDX   *occu;
			KERNELCACHE_IDX   elems;
			KERNELCACHE_IDX   max_elems;
			KERNELCACHE_IDX   time;
			KERNELCACHE_IDX   activenum;
			KERNELCACHE_IDX   buffsize;
			//			LONG   r_offs;
		};

	protected:
		/// kernel cache
		KERNEL_CACHE kernel_cache;

		/// cache_size in MB
		KERNELCACHE_IDX cache_size;

		/// this *COULD* store the whole kernel matrix
		/// usually not applicable / faster
		KERNELCACHE_ELEM* kernel_matrix;

		/// feature vectors to occur on left hand side
		CFeatures* lhs;
		/// feature vectors to occur on right hand side
		CFeatures* rhs;

		REAL combined_kernel_weight ;
	
		bool optimization_initialized ;
		
};
#endif
