"""
Tests for the diagonstics.py file
"""
import pytest
import numpy as np
from turbopy.core import Simulation
from turbopy.diagnostics import FieldDiagnostic


@pytest.fixture(name='simple_field')
def field_fixt(tmp_path):
    """Pytest fixture for FieldDiagnostic class"""
    sim_dic = {"Grid": {"N": 2, "r_min": 0, "r_max": 1},
               "Clock": {"start_time": 0,
                         "end_time": 10,
                         "num_steps": 100},
               "Tools": {"ExampleTool": {}},
               "PhysicsModules": {"ExampleModule": {}},
               "Diagnostics": {
                   "directory": f"{tmp_path}/default_output",
                   # "clock": {},
                   "field": [
                       {"component": "Component",
                        "field": "Field",
                        "output_type": "csv",
                        "filename": "output.csv",
                        "dump_interval": 1}
                       ]
               }
               }
    sim = Simulation(sim_dic)
    sim.read_diagnostics_from_input()
    sim.read_clock_from_input()
    sim.read_grid_from_input()
    field = sim.diagnostics[0]
    return field


# Test methods for FieldDiagnostic class
def test_init_should_create_class_instance_when_called(simple_field):
    """Tests init method in FieldDiagnostic class"""
    assert simple_field.component == "Component"
    assert simple_field.field_name == "Field"
    assert simple_field.output == "csv"
    assert simple_field.field is None
    assert simple_field.dump_interval is None
    assert simple_field.last_dump is None
    assert simple_field.diagnostic_size is None
    assert simple_field.field_was_found is False


def test_check_step_should_update_last_dump_after_dump_interval_has_passed(simple_field):
    """Tests check_step method in FieldDiagnostic class"""
    simple_field.inspect_resource({"Field": np.linspace(0, 1, 2)})
    simple_field.initialize()
    simple_field._owner.clock.time = 1
    simple_field.check_step()
    assert simple_field.last_dump == 1


def test_initialize_should_set_remaining_parameters_when_called(simple_field):
    """Tests initialize method in FieldDiagnostic class for declared attributes"""
    with pytest.raises(RuntimeError):
        assert simple_field.initialize()
    simple_field.inspect_resource({"Field": np.linspace(0, 1, 2)})
    simple_field.initialize()
    assert simple_field.dump_interval == 1
    assert simple_field.last_dump == 0
    assert simple_field.diagnostic_size == (11, 2)
    

def test_initialize_should_set_output_funtion_parameters_when_called(simple_field, tmp_path):
    simple_field.inspect_resource({"Field": np.linspace(0, 1, 2)})
    simple_field.initialize()
    assert simple_field.output_function == simple_field.csv_diagnose
    assert simple_field.csv.filename == f"{tmp_path}/default_output/output.csv"
    assert np.allclose(simple_field.csv.buffer, np.zeros((11, 2)))
    assert simple_field.csv.buffer_index == 0


def test_inspect_resource_should_assign_field_attribute_if_field_name_in_resource(simple_field):
    """Tests inspect_resource method in FieldDiagnostic class"""
    array_to_share = np.linspace(0, 1, 2)
    simple_field.inspect_resource({"Field": array_to_share})
    assert simple_field.field_was_found is True
    assert simple_field.field is array_to_share


def test_csv_diagnose_should_append_data_to_csv_when_called(simple_field):
    """Tests csv_diagnose method in FieldDiagnostic class"""
    simple_field.inspect_resource({"Field": np.linspace(0, 1, 2)})
    simple_field.initialize()
    simple_field.csv.append(simple_field.field)
    assert np.allclose(simple_field.csv.buffer[simple_field.csv.buffer_index - 1, :],
                       simple_field.field)
    assert simple_field.csv.buffer_index == 1
