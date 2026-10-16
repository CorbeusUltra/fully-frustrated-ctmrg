"""Define the sites in the unit cell."""

sites_coords = ((0, 0),(1, 0))


# checkboard

def vertex_to_site(x:int,y:int)->tuple[int,int]:
    return ((x+y)%2,0)

# two_bands
# def vertex_to_site(x:int,y:int)->tuple[int,int]:
#     return (x%2,0)

def tensor_at(x: int, y: int)->int:
    return sites_coords[vertex_to_site(x, y)]
