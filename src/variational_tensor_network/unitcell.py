"""Define the sites in the unit cell."""

type Site = tuple[int, int]



sites_coords: tuple[Site, ...] = ((0, 0),(1, 0))

# checkboard
def unitcell_coord(x:int,y:int)->tuple[int,int]:
    return ((x+y)%2,0)

# # two_bands
# def unitcell_coord(x:int,y:int)->tuple[int,int]:
#     return (x%2,0)


def main():
    print(f"unitcell_coord(2,3)={unitcell_coord(2,3)}")
    print(f"unitcell_coord(3,3)={unitcell_coord(3,3)}")
    # print(f"={}")


if __name__ == "__main__":
    main()
