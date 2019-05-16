import pytest
from parsect.meters.loss_meter import LossMeter


class TestLossMeter:
    def test_loss_basic(self):
        loss = 1
        num_instances = 5
        loss_meter = LossMeter()
        loss_meter.add_loss(avg_batch_loss=loss,
                            num_instances=num_instances)
        assert loss_meter.get_average() == 1

    def test_loss_two_batches(self):
        loss_1 = 1.2
        num_instances_1 = 5
        loss_meter = LossMeter()
        loss_2 = 1.4
        num_instances_2 = 5
        loss_meter.add_loss(avg_batch_loss=loss_1,
                            num_instances=num_instances_1)
        loss_meter.add_loss(avg_batch_loss=loss_2,
                            num_instances=num_instances_2)
        average_loss = loss_meter.get_average()
        assert average_loss == 1.3

