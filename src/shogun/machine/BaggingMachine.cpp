/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/machine/BaggingMachine.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CBaggingMachine::CBaggingMachine()
	: CMachine(),
	m_features(NULL),
	m_num_bags(0),
	m_bag_size(0)
{
	register_parameters();
}

CBaggingMachine::CBaggingMachine(CFeatures* features, CLabels* labels)
	: CMachine(),
	m_num_bags(0),
	m_bag_size(0)
{
	set_labels(labels);
	SG_REF(m_features);
	m_features = features;
	register_parameters();
}

CBaggingMachine::~CBaggingMachine()
{
	if (m_machine) {
		SG_UNREF(m_machine);
		m_machine = NULL;
	}

	if (m_features) {
		SG_UNREF(m_features);
	}
}

CLabels* CBaggingMachine::apply(CFeatures* data)
{
	CLabels* pred = NULL;

	for (int32_t i = 0; i < m_bags.get_num_elements(); ++i)
	{
		CMachine* m = dynamic_cast<CMachine*>(m_bags.get_element(i));
		CLabels* l = m->apply(data);

		// voting scheme is required

		SG_UNREF(l);
	}

	return pred;
}

CRegressionLabels* CBaggingMachine::apply_regression(CFeatures* data)
{
	SGVector<float64_t> avg_labels;
	avg_labels.zero();

	for (int32_t i = 0; i < m_bags.get_num_elements(); ++i)
	{
		CMachine* m = dynamic_cast<CMachine*>(m_bags.get_element(i));
		CRegressionLabels* l = m->apply_regression(data);
		SGVector<float64_t> lv = l->get_values();

		avg_labels.add(lv);

		SG_UNREF(l);
	}

	float64_t scale = 1/(float64_t)m_bags.get_num_elements();
	avg_labels.scale(scale);

	CRegressionLabels* pred = new CRegressionLabels(avg_labels);

	return pred;
}

bool CBaggingMachine::train_machine(CFeatures* data)
{
	REQUIRE(m_machine != NULL, "Machine type is not set!");
	REQUIRE(m_bag_size > 0, "Bag size is not set!");
	REQUIRE(m_num_bags > 0, "Number of bag is not set!");

	for (int32_t i = 0; i < m_num_bags; ++i)
	{
		CMachine* c = dynamic_cast<CMachine*>(m_machine->clone());
		SGVector<index_t> idx(m_bag_size);
		idx.random(0, m_features->get_num_vectors());
		m_features->add_subset(idx);
		c->set_labels(m_labels);
		c->train(m_features);
		m_features->remove_subset();
		m_bags.append_element(c);
	}

	return true;
}

void CBaggingMachine::register_parameters()
{
	SG_ADD((CSGObject**)&m_features, "features", "Train features for bagging", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_bags, "num_bags", "Number of bags", MS_AVAILABLE);
	SG_ADD(&m_bag_size, "bag_size", "Number of vectors per bag", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_bags, "bags", "Bags array", MS_NOT_AVAILABLE);
}

void CBaggingMachine::set_num_bags(int32_t num_bags)
{
	m_num_bags = num_bags;
}

int32_t CBaggingMachine::get_num_bags() const
{
	return m_num_bags;
}

void CBaggingMachine::set_bag_size(int32_t bag_size)
{
	m_bag_size = bag_size;
}

int32_t CBaggingMachine::get_bag_size() const
{
	return m_bag_size;
}

CMachine* CBaggingMachine::get_machine() const
{
	SG_REF(m_machine);
	return m_machine;
}

void CBaggingMachine::set_machine(CMachine* machine)
{
	if (m_machine) {
		SG_UNREF(m_machine);
		m_machine = NULL;
	}

	SG_REF(machine);
	m_machine = machine;
}
