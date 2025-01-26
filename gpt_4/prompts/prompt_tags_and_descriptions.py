prompt_tags = """
Detect all objects in the picture, generate a description for each object, and decide whether it is floor-object, wall-object.

Here is the definations of the three kind:
floor object: object that is placed on floor or in direct contact with the floor.
wall object: object that is placed on wall and not in contact with the floor.

Here is a sample answer:
table: A big yellow table | floor-object
chair: A gray armchair | floor-object
tv: A black wall-mounted television | wall-object

Requirements:
1. Description should not be too long.
2. You should only give the result and no unnecessary words.
3. Don't describe the positional relationship between objects.
4. Classification can only be **floor-object**, **wall-object**.
5. Please pay attention to only large furniture, and ignore small objects like bottles or books.
"""

