/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#ifndef BAGGINGMACHINE_H
#define BAGGINGMACHINE_H

#include <shogun/machine/Machine.h>

namespace shogun
{
	/**
	 * @brief: Bagging algorithm
	 * i.e. bootstrap aggregating
     */
	class CBaggingMachine : public CMachine
	{
		public:
			/** default ctor */
			CBaggingMachine();

			/**
			 * constructor
			 *
			 * @param features training features
			 * @param labals training labels
			 */
			CBaggingMachine(CFeatures* features, CLabels* labels);

			virtual ~CBaggingMachine();

			virtual CLabels* apply(CFeatures* data=NULL);
			virtual CRegressionLabels* apply_regression(CFeatures* data=NULL);

			/**
			 * Set number of bags/machine to create
			 *
			 * @param num_bags number of bags
			 */
			void set_num_bags(int32_t num_bags);

			/**
			 * Get number of bags/machines
			 *
			 * @return number of bags
			 */
			int32_t get_num_bags() const;

			/**
			 * Set number of feature vectors to use
			 * for each bag/machine
			 *
			 * @param bag_size number of vectors to use for a bag
			 */
			void set_bag_size(int32_t bag_size);

			/**
			 * Get number of feature vectors that are use
			 * for training each bag/machine
			 *
			 * @return number of vectors used for training for each bag.
			 */
			int32_t get_bag_size() const;

			/**
			 * Get machine for bagging
			 *
			 * @return machine that is being used in bagging
			 */
			CMachine* get_machine() const;

			/**
			 * Set machine to use in bagging
			 *
			 * @param machine the machine to use for bagging
			 */
			void set_machine(CMachine* machine);
			
			/** name **/
			virtual const char* get_name() const { return "BaggingMachine"; }

		protected:
			virtual bool train_machine(CFeatures* data=NULL);

		private:
			void register_parameters();
			void init();

		private:
			/** bags array */
			CDynamicObjectArray m_bags;

			/** features to train on */
			CFeatures* m_features;

			/** machine to use for bagging */
			CMachine* m_machine;

			/** number of bags to create */
			int32_t m_num_bags;

			/** number of vectors to use from the training features */
			int32_t m_bag_size;
	};
}

#endif /* BAGGINGMACHINE_H */
