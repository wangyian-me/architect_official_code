prompt_inpainting_prompt_generation = """
Given the objects in the {current_scene}, please list which objects have already reached their potential limits, and the objects are still lacking. 
Your answer should be in the following format: 
reached limit: object A, object B, ...
lacking: object C, object D, ...

The objects in the {current_scene} are: {counting_objects} 

Remember, do not answer anything not asked. The lacking objects should ideally contain objects that are not in the {current_scene}. The lacking objects you list should be precise, do not give things like "other furniture" and do not give the amount of objects.
"""