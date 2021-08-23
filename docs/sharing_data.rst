Sharing Resources
=================

It is often necessary to share resources between custom :class:`turbopy.core.PhyiscsModules` or :class:`turbopy.core.Diagnostics`. A new API has been developed to assist with this. To use this new API, you simply need to define a couple of dictionaries (``_resources_to_share`` and ``_needed_resources``) in your class. Then, when the ``prepare_simulation`` method of your simulation is called, the shared variables get set up automatically.

Making resources available to other modules
-------------------------------------------

In order to tell other :class:`turbopy.core.PhyiscsModules` about resources that you want to share, just add them to the member variable ``_resources_to_share`` in the ``__init__`` method. For example, the following function will share the variables ``self.position`` and ``self.momentum``::

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self.position = np.zeros((1, 3))
        self.momentum = np.zeros((1, 3))

        self._resources_to_share = {"position": self.position,
                                    "momentum": self.momentum}

Note that the variables that you want to share need to be defined before they can be added to the ``_resources_to_share`` dictionary. Also, make sure that they are *mutable* variables, otherwise other modules won't see any changes that you make to them during the simulation.


Looking for shared resources
----------------------------

If your module needs access to a variable that is being shared from a different module, you use the member variable ``_needed_resources``. In this example, the data shared by the example above will be saved. ::

    def __init__(self, owner: Simulation, input_data: dict):
        super().__init__(owner, input_data)
        self._needed_resources = {"position": "x",
                                  "momentum": "p"}

This will create the variables ``self.x`` and ``self.p``, which will point to the position and momentum data shared by the second module.
