from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

comm_group=comm.Split(color=rank/4,key=rank)
size_group=comm_group.Get_size()
rank_group=comm_group.Get_rank()

print "WORLD RANK/SIZE: %d/%d \t ROW RANK/SIZE: %d/%d\n"%(rank, size, rank_group, size_group)
    
