import click
import os
import rich


@click.command(help="Peek inside a .h5smu file")
@click.argument("path", default=False, type=str)
def peek(path):
    if not os.path.isfile(path):
        print('Please specify a valid SpatialMuon file. Example "spatialmuon peek my_data.h5smu"')
    else:
        from spatialmuon import SpatialMuData

        smu = SpatialMuData(backing=path, backingmode="r")
        print(smu)


@click.group()
def cli():
    pass


cli.add_command(peek)


def main():
    cli()
